#if 0
#include "diffusers/pipelines/flux2/Flux2KleinPipeline.hpp"
#include "nn/RethrowVisitor.hpp"
#include "ggml/GGUFLoaderVisitor.hpp"
#include "ggml/Runtime.hpp"
#include "ggml/Computation.hpp"
#include "transformers/models/qwen3/Qwen3Config.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <utility>

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

// Mirrors diffusers `calculate_shift` for FlowMatchEulerDiscreteScheduler
// with use_dynamic_shifting.
static float calculate_shift(float image_seq_len, float base_seq_len, float max_seq_len,
                      float base_shift, float max_shift) {
    const float m = (max_shift - base_shift) / (max_seq_len - base_seq_len);
    const float b = base_shift - m * base_seq_len;
    return m * image_seq_len + b;
}

std::pair<Tensor, Tensor>
Flux2KleinPipeline::encode_prompt(Runtime& runtime, const std::string& prompt, const GenerationOptions& options) {
    auto tokenized = tokenizer_.encode(prompt, options.max_sequence_length, false);
    
    auto input_ids = runtime.create<int32_t>({(int64_t)tokenized.input_ids.size(), 1},
        [=](Tensor, std::mt19937&) {
            return tokenized.input_ids;
        });

    // Upload attention mask as float directly for masking operations
    auto attention_mask = runtime.create<float>({(int64_t)tokenized.attention_mask.size(), 1},
        [=](Tensor, std::mt19937&) {
            return std::move(std::vector<float>(tokenized.attention_mask.begin(), tokenized.attention_mask.end()));
        });

    // Pass attention_mask to text encoder 
    auto prompt_embeds = text_encoder_.forward(runtime, input_ids, attention_mask);
    
    // Masked mean pooling over sequence dimension (axis=1)
    // prompt_embeds: (hidden, seq, B)
    // attention_mask: (seq, 1) - needs broadcast to (1, seq, B)
    auto mask_expanded = attention_mask.unsqueeze(/*axis=*/0);  // (1, seq, 1)
    auto masked_embeds = prompt_embeds * mask_expanded;
    auto summed = masked_embeds.sum(/*axis=*/1);  // (hidden, 1, B)

    auto divisor = tokenized.num_real_tokens > 0 ? (float)tokenized.num_real_tokens : 1.0f;
    auto pooled = summed / divisor;

    // One graph, two outputs
    return {prompt_embeds, pooled};
}

Tensor Flux2KleinPipeline::prepare_latents(Runtime& runtime, int batch, int packed_h, int packed_w) {
    const int64_t token_dim = int64_t(config_.num_latent_channels) * 4; // C * 2 * 2
    const size_t count = size_t(batch) * packed_h * packed_w * token_dim;

    // Sampling noise directly in packed layout is equivalent to diffusers'
    // randn(B, C, 2h, 2w) + _pack_latents — packing is a pure permutation.
    return runtime.create<float>({token_dim, int64_t(packed_h) * packed_w, batch},
        [=](Tensor, std::mt19937& rng) {
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
Tensor Flux2KleinPipeline::unpack_latents(Tensor packed, int channels, int packed_h, int packed_w) {
    const int64_t C = channels;
    const int64_t B = packed.shape()[2];

    Tensor t = packed.reshape({2 * C, 2, packed_w, packed_h * B});
    t = t.permute(/*axes=*/{0, 2, 1, 3});   // swap p1 <-> x
    t = t.contiguous();
    return t.reshape({C, 2 * packed_w, 2 * packed_h, B});
}

Tensor Flux2KleinPipeline::prepare_img_ids(Runtime& runtime, int packed_h, int packed_w) {
    return runtime.create<float>({3, int64_t(packed_h) * packed_w},
        [=](Tensor, std::mt19937&) {
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

Tensor Flux2KleinPipeline::prepare_txt_ids(Runtime& runtime, int seq_len) {
    return Tensor::zeros(*runtime.context(), {3, seq_len});
}

void Flux2KleinPipeline::prepare_timesteps(PipelineState& state,
                                           const GenerationOptions& options) {
    float mu = 0.0f;
    if (config_.use_dynamic_shifting) {
        const int image_seq_len = (state.latent_height / 2) * (state.latent_width / 2);
        mu = calculate_shift((float)image_seq_len,
                             (float)config_.base_seq_len, (float)config_.max_seq_len,
                             config_.base_shift, config_.max_shift);
    }
    scheduler_.set_timesteps(options.num_inference_steps, mu);
    state.timesteps = scheduler_.timesteps();
    state.sigmas = scheduler_.sigmas();
}

Tensor Flux2KleinPipeline::repeat_batch(const Tensor& t, int batch) {
    if (batch <= 1) return t;
    Runtime rt;
    auto shape = t.shape();
    shape.back() = batch;                    // batch is the slowest dim (ne[ndim-1])
    return executor_.execute(rt, t.repeat(shape));   // persistent result
}

Flux2KleinPipeline::PipelineState Flux2KleinPipeline::prepare(Runtime& runtime, const std::string& prompt, const GenerationOptions& options) {
    PipelineState state;

    if (options.height % config_.vae_scale_factor != 0 ||
        options.width % config_.vae_scale_factor != 0)
        throw std::runtime_error("height/width must be divisible by vae_scale_factor");

    // diffusers: height = 2 * (H // vae_scale_factor) inside prepare_latents
    state.latent_height = 2 * (options.height / config_.vae_scale_factor);
    state.latent_width  = 2 * (options.width  / config_.vae_scale_factor);
    const int packed_h = state.latent_height / 2;
    const int packed_w = state.latent_width / 2;

    // Build once; from here on only `latents` mutates (Phase 2).
    auto [embeds, pooled] = encode_prompt(runtime, prompt, options);
    state.prompt_embeds        = repeat_batch(embeds, options.num_images_per_prompt);
    state.pooled_prompt_embeds = repeat_batch(pooled, options.num_images_per_prompt);

    state.latents = prepare_latents(runtime, options.num_images_per_prompt, packed_h, packed_w);
    state.img_ids = prepare_img_ids(runtime, packed_h, packed_w);
    state.txt_ids = prepare_txt_ids(runtime, options.max_sequence_length);

    prepare_timesteps(state, options);
    return state;
}

Tensor Flux2KleinPipeline::build_step(Runtime& rt, const PipelineState& state,
                                      float timestep) {
    Tensor t = Tensor::full(*rt.context(), {1}, timestep);
    Tensor guidance = Tensor::full(*rt.context(), {1}, 0.0f); // TODO: check if config.guidance_embeds is true

    Tensor noise_pred = transformer_.forward(rt,
        /*hidden_states=*/          state.latents,
        /*timestep=*/               t,
        /*guidance=*/               guidance,
        /*encoder_hidden_states=*/  state.prompt_embeds,
        /*pooled_projections=*/     state.pooled_prompt_embeds,
        /*img_ids=*/                state.img_ids,
        /*txt_ids=*/                state.txt_ids);

    // Bakes dt = sigma_next - sigma into the graph; advances step_index.
    return scheduler_.step(rt, noise_pred, state.latents);
}

Tensor Flux2KleinPipeline::build_decode(Runtime& rt, const PipelineState& state,
                                        const GenerationOptions& options) {
    Tensor z = unpack_latents(rt, state.latents, config_.num_latent_channels,
                              state.latent_height / 2, state.latent_width / 2);

    // diffusers: z = z / vae.scaling_factor + vae.shift_factor
    z = z / config_.vae_scaling_factor;
    z = z + config_.vae_shift_factor;

    return vae_.decode(rt, z);   // (W, H, 3, B)
}

// decoded: ne (W, H, 3, B), float, contiguous; converts CHW planes -> HWC uint8
std::vector<Image> Flux2KleinPipeline::latents_to_images(const Tensor& decoded, int batch, int height, int width) {
    std::vector<float> data = executor_.read<float>(decoded);

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

std::vector<Image> Flux2KleinPipeline::generate(Runtime& runtime, 
                                                const std::string& prompt,
                                                const GenerationOptions& options) {
    const auto t_start = std::chrono::steady_clock::now();

    PipelineState state = prepare(runtime, prompt, options);

    // denoising loop: a NEW graph every iteration.
    for (size_t i = 0; i < state.timesteps.size(); ++i) {
        Tensor next = build_step(runtime, state, state.timesteps[i]);
        Computation computation(next);

        const auto t0 = std::chrono::steady_clock::now();

        state.latents = executor_.execute(runtime, next);              // sync point

        // `next`'s nodes lived in `rt`'s arena; execute() handed back
        // a persistent copy, so `rt` may die at the end of this iteration.

        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        std::cerr << "[flux2-klein] step " << (i + 1) << "/" << state.timesteps.size()
                  << "  t=" << state.timesteps[i] << "  (" << ms << " ms)\n";
    }

    // decode: same pattern, one graph, one sync point.
    std::vector<Image> images;
    {
        Tensor decoded = build_decode(runtime, state, options);
        decoded = executor_.execute(runtime, decoded);
        images = latents_to_images(decoded, options.num_images_per_prompt,
                                   options.height, options.width);
    }

    const double total = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_start).count();
    std::cerr << "[flux2-klein] done (" << total << " ms)\n";
    return images;
}
#endif