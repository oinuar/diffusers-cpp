#include "diffusers/pipelines/flux2/Flux2KleinPipeline.hpp"
#include "nn/RethrowVisitor.hpp"
#include "ggml/GGUFLoaderVisitor.hpp"
#include "ggml/Runtime.hpp"
#include "ggml/Scheduler.hpp"
#include "ggml/Computation.hpp"
#include "transformers/models/qwen3/Qwen3Config.hpp"
#include "ProgressBar.hpp"
#include <iostream>

Flux2KleinPipeline Flux2KleinPipeline::from_pretrained(Backend& loader_backend, const std::filesystem::path& path) {
    // 1. Initialize models from config
    Qwen2TokenizerFast tokenizer(path / "tokenizer" / "config.json");
    Qwen3ForCausalLM text_encoder(Qwen3Config::from_file(path / "text_encoder" / "config.json"));
    Flux2Transformer2DModel transformer(Flux2Transformer2DModel::Config::from_file(path / "transformer" / "config.json"));
    AutoencoderKLFlux2 vae(AutoencoderKLFlux2::Config::from_file(path / "vae" / "config.json"));

    // 2. Access GGUF models
    GGUFLoaderVisitor vae_loader(loader_backend, path / "vae");
    GGUFLoaderVisitor text_encoder_loader(loader_backend, path / "text_encoder");
    GGUFLoaderVisitor transformer_loader(loader_backend, path / "transformer");

    // 3. Load model structure
    RethrowVisitor vae_visitor(vae_loader);
    RethrowVisitor text_encoder_visitor(text_encoder_loader);
    RethrowVisitor transformer_visitor(transformer_loader);
    vae.accept(vae_visitor);
    vae_visitor.rethrow();
    text_encoder.accept(text_encoder_visitor);
    text_encoder_visitor.rethrow();
    transformer.accept(transformer_visitor);
    transformer_visitor.rethrow();

    // 4. Construct the pipeline
    Flux2KleinPipeline pipeline(
        std::move(transformer),
        std::move(vae),
        std::move(text_encoder),
        std::move(tokenizer)
    );
    
    return std::move(pipeline);
}

static Tensor repeat_batch(const Tensor& t, int batch) {
    if (batch <= 1) return t;

    auto shape = t.shape();
    shape[-1] = batch;                    // batch is the slowest dim (ne[ndim-1])
    return t.repeat(shape);
}

std::pair<Tensor, Tensor>
Flux2KleinPipeline::encode_prompt(Runtime& runtime, const std::string& prompt, size_t max_sequence_length) {
    std::vector<int> mask;
    size_t num_real_tokens;

    auto tokens = tokenizer_.encode(prompt, max_sequence_length, false, &mask, &num_real_tokens);
    
    auto input_ids = runtime.create<int32_t>({(int64_t)tokens.size(), 1},
        [=](std::mt19937&) {
            return std::move(tokens);
        });

    // Upload attention mask as float directly for masking operations
    auto attention_mask = runtime.create<float>({(int64_t)mask.size(), 1},
        [=](std::mt19937&) {
            return std::move(std::vector<float>(mask.begin(), mask.end()));
        });

    // Pass attention_mask to text encoder 
    auto prompt_embeds = text_encoder_.forward(runtime, input_ids, attention_mask);
    
    // Masked mean pooling over sequence dimension (axis=1)
    // prompt_embeds: (hidden, seq, B)
    // attention_mask: (seq, 1) - needs broadcast to (1, seq, B)
    auto mask_expanded = attention_mask.unsqueeze(/*axis=*/0);  // (1, seq, 1)
    auto masked_embeds = prompt_embeds * mask_expanded;
    auto summed = masked_embeds.sum(/*axis=*/1);  // (hidden, 1, B)

    auto divisor = num_real_tokens > 0 ? (float)num_real_tokens : 1.0f;
    auto pooled = summed / divisor;

    // One graph, two outputs
    return {prompt_embeds, pooled};
}

static Tensor prepare_img_ids(Runtime& runtime, int packed_h, int packed_w) {
    return runtime.create<float>({3, int64_t(packed_h) * packed_w},
        [=](std::mt19937&) {
            std::vector<float> ids(size_t(packed_h) * packed_w * 3, 0.0f);

            for (int y = 0; y < packed_h; ++y)
                for (int x = 0; x < packed_w; ++x) {
                    float* row = &ids[(size_t(y) * packed_w + x) * 3];
                    row[0] = 0.0f;           // diffusers: zeros, then row/col indices
                    row[1] = (float)y;
                    row[2] = (float)x;
                }

            return std::move(ids);
        });
}

static Tensor prepare_txt_ids(Runtime& runtime, int seq_len) {
    return Tensor::zeros(*runtime.context(), {3, seq_len});
}

// Mirrors diffusers `calculate_shift` for FlowMatchEulerDiscreteScheduler
// with use_dynamic_shifting.
static float calculate_shift(float image_seq_len, float base_seq_len, float max_seq_len,
                      float base_shift, float max_shift) {
    const float m = (max_shift - base_shift) / (max_seq_len - base_seq_len);
    const float b = base_shift - m * base_seq_len;
    return m * image_seq_len + b;
}

static Tensor noisy_latents(Runtime& runtime, int batch, int packed_h, int packed_w) {
    const int64_t token_dim = int64_t(config_.num_latent_channels) * 4; // C * 2 * 2
    const size_t count = size_t(batch) * packed_h * packed_w * token_dim;

    // Sampling noise directly in packed layout is equivalent to diffusers'
    // randn(B, C, 2h, 2w) + _pack_latents — packing is a pure permutation.
    return runtime.create<float>({token_dim, int64_t(packed_h) * packed_w, batch},
        [=](std::mt19937& rng) {
            std::vector<float> noise(count);
            std::normal_distribution<float> normal(0.0f, 1.0f);
            for (float& v : noise) v = normal(rng);

            return std::move(noise);
        });
}

// Graph-native 2x2 unpack, pure 4D.
//
//   packed: ne (4C, N, B), N = packed_h * packed_w
//   result: ne (2pw, 2ph, C, B)   (torch: (B, C, 2ph, 2pw))
//
static Tensor unpack_latents(Tensor packed, int channels, int packed_h, int packed_w) {
    const int64_t C = channels;
    const int64_t B = packed.shape()[2];

    Tensor t = packed.reshape({2 * C, 2, packed_w, packed_h * B});
    t = t.permute(/*axes=*/{0, 2, 1, 3});   // swap p1 <-> x
    t = t.contiguous();
    return t.reshape({C, 2 * packed_w, 2 * packed_h, B});
}

// decoded: ne (W, H, 3, B), float, contiguous; converts CHW planes -> HWC uint8
static std::vector<Image> latents_to_images(std::vector<float>&& data, int batch, int height, int width) {
    std::vector<Image> images;
    images.reserve(batch);

    const size_t w = (size_t)width;
    const size_t h = (size_t)height;
    const size_t plane = w * h;                       // per channel
    constexpr size_t channels = 3;

    for (int b = 0; b < batch; ++b) {
        std::vector<uint8_t> pixels(plane * channels);
        const float* src = data.data() + size_t(b) * channels * plane;

        for (size_t y = 0; y < h; ++y)
            for (size_t x = 0; x < w; ++x)
                for (size_t c = 0; c < channels; ++c) {
                    float v = src[c * plane + y * w + x];          // CHW source
                    v = std::clamp(v * 0.5f + 0.5f, 0.0f, 1.0f);
                    pixels[(y * w + x) * channels + c] =           // HWC destination
                        static_cast<uint8_t>(std::lrint(v * 255.0f));
                }

        images.emplace_back(w, h, channels, std::move(pixels));
    }

    return images;
}

std::vector<Image> Flux2KleinPipeline::generate(Scheduler& scheduler, const GenerationOptions& options) {
    if (options.height % config_.vae_scale_factor != 0 ||
        options.width % config_.vae_scale_factor != 0)
        throw std::runtime_error("height/width must be divisible by vae_scale_factor");

    // diffusers: height = 2 * (H // vae_scale_factor) inside prepare_latents
    auto latent_height = 2 * (options.height / config_.vae_scale_factor);
    auto latent_width  = 2 * (options.width  / config_.vae_scale_factor);

    const int packed_h = latent_height / 2;
    const int packed_w = latent_width / 2;

    const auto t_start = std::chrono::steady_clock::now();

    // Embeds
    std::vector<float> prompt_embeds_data, pooled_prompt_embeds_data, img_ids_data;
    std::vector<int32_t> txt_ids_data;
    Tensor::Shape prompt_embeds_shape, pooled_prompt_embeds_shape, img_ids_shape, txt_ids_shape;
    {
        Runtime runtime(scheduler);
        
        auto [embeds, pooled] = encode_prompt(runtime, options.prompt, options.max_sequence_length);
        auto prompt_embeds = repeat_batch(embeds, options.num_images_per_prompt);
        auto pooled_prompt_embeds = repeat_batch(pooled, options.num_images_per_prompt);

        auto img_ids = prepare_img_ids(runtime, packed_h, packed_w);
        auto txt_ids = prepare_txt_ids(runtime, options.max_sequence_length);

        Graph graph(runtime, {
            prompt_embeds,          // (hidden, seq, B
            pooled_prompt_embeds,   // (hidden, 1, B)
            img_ids,                // (3, N_img)
            txt_ids                 // (3, seq)
        });

        Computation computation(graph);

        prompt_embeds_data = std::move(computation.read<float>(prompt_embeds));
        prompt_embeds_shape = prompt_embeds.shape();
        pooled_prompt_embeds_data = std::move(computation.read<float>(pooled_prompt_embeds));
        pooled_prompt_embeds_shape = pooled_prompt_embeds.shape();
        img_ids_data = std::move(computation.read<float>(img_ids));
        img_ids_shape = img_ids.shape();
        txt_ids_data = std::move(computation.read<int32_t>(txt_ids));
        txt_ids_shape = txt_ids.shape();
    }

    // Setup timesteps
    {
        auto mu = 0.0f;

        if (config_.use_dynamic_shifting) {
            auto image_seq_len = (latent_height / 2) * (latent_width / 2);
            mu = calculate_shift((float)image_seq_len,
                                (float)config_.base_seq_len, (float)config_.max_seq_len,
                                config_.base_shift, config_.max_shift);
        }

        scheduler_.set_timesteps(options.num_inference_steps, mu);
    }

    // Denoising
    std::vector<float> latents_data;
    Tensor::Shape latents_shape;
    {
        Runtime runtime(scheduler);

        auto prompt_embeds = runtime.create<float>(prompt_embeds_shape,
            [&](std::mt19937&) {
                return std::move(prompt_embeds_data);
            });

        auto pooled_prompt_embeds = runtime.create<float>(pooled_prompt_embeds_shape,
            [&](std::mt19937&) {
                return std::move(pooled_prompt_embeds_data);
            });

        auto img_ids = runtime.create<float>(img_ids_shape,
            [&](std::mt19937&) {
                return std::move(img_ids_data);
            });

        auto txt_ids = runtime.create<int32_t>(txt_ids_shape,
            [&](std::mt19937&) {
                return std::move(txt_ids_data);
            });

        auto latents = noisy_latents(runtime, options.num_images_per_prompt, packed_h, packed_w);

        float current_timestep;
        auto timestep = runtime.create<float>({1}, [&](std::mt19937&) {
            return std::vector<float>({current_timestep});
        });

        auto guidance = Tensor::zeros(*runtime.context(), {1}); // TODO: check if config.guidance_embeds is true

        auto noise_pred = transformer_.forward(runtime,
            /*hidden_states=*/          latents,
            /*encoder_hidden_states=*/  prompt_embeds,
            /*timestep=*/               timestep,
            /*img_ids=*/                img_ids,
            /*txt_ids=*/                txt_ids,
            /*guidance=*/               guidance,
            /*pooled_projections=*/     pooled_prompt_embeds /* TODO: where is this? */);

        // Bakes dt = sigma_next - sigma into the graph; advances step_index.    
        auto dt = runtime.create<float>({1}, [&](std::mt19937&) {
            return std::vector<float>({scheduler_.step()}); // we must step during the computation
        });

        // Integrate latents over dt and assign latents <- latents+1
        auto next_latents = scheduler_.integrate(noise_pred, latents, dt);
        next_latents = next_latents.copy(latents);

        Graph denoise(runtime, {next_latents});
        ProgressBar progress("Denoising", scheduler_.get_timesteps().size());

        for (auto i = 0; i < scheduler_.get_timesteps().size(); ++i) {
            const auto t0 = std::chrono::steady_clock::now();

            current_timestep = scheduler_.get_timesteps()[i];

            Computation computation(denoise);

            // unbind tensors from initialization because
            // values are carried over by the computation now
            runtime.unbind(prompt_embeds);
            runtime.unbind(pooled_prompt_embeds);
            runtime.unbind(img_ids);
            runtime.unbind(txt_ids);
            runtime.unbind(latents);

            if (i == scheduler_.get_timesteps().size() - 1) {
                latents_data = computation.read<float>(latents);
                latents_shape = latents.shape();
            }

            progress.update(i);
        }
    }

    // Decode
    std::vector<Image> images;
    {
        Runtime runtime(scheduler);

        auto latents = runtime.create<float>(latents_shape, [&](std::mt19937&) {
            return std::move(latents_data);
        });

        auto z = unpack_latents(latents, config_.num_latent_channels, packed_h, packed_w);

        // diffusers: z = z / vae.scaling_factor + vae.shift_factor
        z = z / config_.vae_scaling_factor;
        z = z + config_.vae_shift_factor;

        auto decoded = vae_.decode(runtime, z);   // (W, H, 3, B)

        Graph graph(runtime, {decoded});
        Computation computation(graph);

        auto data = computation.read<float>(decoded);

        images = latents_to_images(std::move(data), options.num_images_per_prompt, options.height, options.width);
    }

    return std::move(images);
}
