#include "diffusers/pipelines/flux2/Flux2KleinPipeline.hpp"
#include "nn/RethrowVisitor.hpp"
#include "nn/Parameter.hpp"
#include "nn/modules/BatchNorm2d.hpp"
#include "ggml/GGUFLoaderVisitor.hpp"
#include "ggml/Runtime.hpp"
#include "ggml/Scheduler.hpp"
#include "ggml/Computation.hpp"
#include "transformers/models/qwen3/Qwen3Config.hpp"
#include "ProgressBar.hpp"
#include <vector>
#include <string>
#include <random>
#include <filesystem>
#include <stdexcept>
#include <cmath>
#include <chrono>
#include <optional>

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
    shape[0] = batch;
    return t.repeat(shape);
}

std::pair<Tensor, Tensor>
Flux2KleinPipeline::encode_prompt(Runtime& runtime, int batch, const std::string& prompt, size_t max_sequence_length) {
    std::vector<int> mask;
    size_t num_real_tokens;

    // Apply chat template to prompt
    std::string text = tokenizer_.apply_chat_template(
        {{"user", prompt}},
        /*add_generation_prompt=*/true,
        /*enable_thinking=*/false
    );

    auto tokens = tokenizer_.encode(text, max_sequence_length, /*add_special_tokens=*/false, &mask, &num_real_tokens);
    
    auto input_ids = runtime.create<int32_t>({batch, (int64_t)tokens.size()},
        [=](std::mt19937&) {
            std::vector<int32_t> ids(size_t(batch) * tokens.size());
            for (auto b=0; b<batch; ++b) std::copy(tokens.begin(), tokens.end(), ids.begin() + b * tokens.size());
            return std::move(ids);
        });

    auto attention_mask = runtime.create<float>({batch, (int64_t)mask.size()},
        [=](std::mt19937&) {
            std::vector<float> m(size_t(batch) * mask.size());
            for (auto b=0; b<batch; ++b) std::copy(mask.begin(), mask.end(), m.begin() + b * mask.size());
            return std::move(m);
        });

    // Extract hidden states from specific layers
    std::unordered_map<size_t, Tensor> hidden_states = {
        {9, Tensor()},
        {18, Tensor()},
        {27, Tensor()},
    };

    text_encoder_.forward(
        runtime,
        input_ids,
        attention_mask,
        std::nullopt, // position_ids
        std::nullopt, // past_key_values
        std::nullopt, // inputs_embeds
        std::nullopt, // labels
        std::nullopt, // use_cache
        0, // logits_to_keep
        &hidden_states
    );

    auto l9 = hidden_states[9];
    auto l18 = hidden_states[18];
    auto l27 = hidden_states[27];

    // Stack along a new dimension (axis=1) -> (B, 3, seq, hidden)
    auto stacked = Tensor::stack({l9, l18, l27}, /*axis=*/1);
    
    // Permute to (B, seq, 3, hidden)
    auto permuted = stacked.permute({0, 2, 1, 3});
    
    // Reshape to (B, seq, 3 * hidden)
    int64_t seq_len = tokens.size();
    int64_t hidden_dim = l9.shape()[2]; // Assuming (B, seq, hidden)
    auto prompt_embeds = permuted.reshape({batch, seq_len, 3 * hidden_dim});

    // Generate 4D txt_ids: (B, seq_len, 4) -> [0, 0, 0, L]
    auto txt_ids = runtime.create<float>({batch, seq_len, 4},
        [=](std::mt19937&) {
            std::vector<float> ids(size_t(batch) * seq_len * 4, 0.0f);
            for (int b = 0; b < batch; ++b) {
                for (int l = 0; l < seq_len; ++l) {
                    float* row = &ids[(size_t(b) * seq_len + l) * 4];
                    row[0] = 0.0f; // T
                    row[1] = 0.0f; // H
                    row[2] = 0.0f; // W
                    row[3] = (float)l; // L
                }
            }
            return std::move(ids);
        });

    return {prompt_embeds, txt_ids};
}

// 4D img_ids: (B, N, 4) -> [0, y, x, 0]
static Tensor prepare_img_ids(Runtime& runtime, int batch, int packed_h, int packed_w) {
    int64_t N = int64_t(packed_h) * packed_w;
    return runtime.create<float>({batch, N, 4},
        [=](std::mt19937&) {
            std::vector<float> ids(size_t(batch) * N * 4, 0.0f);
            for (int b = 0; b < batch; ++b) {
                for (int y = 0; y < packed_h; ++y) {
                    for (int x = 0; x < packed_w; ++x) {
                        float* row = &ids[(size_t(b) * N + y * packed_w + x) * 4];
                        row[0] = 0.0f; // T
                        row[1] = (float)y; // H
                        row[2] = (float)x; // W
                        row[3] = 0.0f; // L
                    }
                }
            }
            return std::move(ids);
        });
}

// 4D image_latent_ids for img2img: (B, N_ref, 4) -> [scale + i*scale, y, x, 0]
static Tensor prepare_image_ids(Runtime& runtime, int batch, int num_images, int packed_h, int packed_w) {
    int64_t N = int64_t(packed_h) * packed_w;
    int scale = 10;
    return runtime.create<float>({batch, int64_t(num_images) * N, 4},
        [=](std::mt19937&) {
            std::vector<float> ids(size_t(batch) * num_images * N * 4, 0.0f);
            for (int b = 0; b < batch; ++b) {
                for (int i = 0; i < num_images; ++i) {
                    float t = scale + scale * i;
                    for (int y = 0; y < packed_h; ++y) {
                        for (int x = 0; x < packed_w; ++x) {
                            float* row = &ids[(size_t(b) * num_images * N + i * N + y * packed_w + x) * 4];
                            row[0] = t;
                            row[1] = (float)y;
                            row[2] = (float)x;
                            row[3] = 0.0f;
                        }
                    }
                }
            }
            return std::move(ids);
        });
}

// Python compute_empirical_mu
static float compute_empirical_mu(int64_t image_seq_len, int num_steps) {
    const float a1 = 8.73809524e-05f, b1 = 1.89833333f;
    const float a2 = 0.00016927f, b2 = 0.45666666f;
    if (image_seq_len > 4300) {
        return a2 * (float)image_seq_len + b2;
    }
    float m_200 = a2 * (float)image_seq_len + b2;
    float m_10 = a1 * (float)image_seq_len + b1;
    float a = (m_200 - m_10) / 190.0f;
    float b = m_200 - 200.0f * a;
    return a * (float)num_steps + b;
}

static Tensor noisy_latents(Runtime& runtime, int batch, int packed_h, int packed_w, int num_latent_channels) {
    const int64_t token_dim = int64_t(num_latent_channels) * 4; // C * 4
    const size_t count = size_t(batch) * packed_h * packed_w * token_dim;

    return runtime.create<float>({batch, int64_t(packed_h) * packed_w, token_dim},
        [=](std::mt19937& rng) {
            std::vector<float> noise(count);
            std::normal_distribution<float> normal(0.0f, 1.0f);
            for (float& v : noise) v = normal(rng);
            return std::move(noise);
        });
}

// Graph-native 2x2 unpack.
//
// Input:
//   packed: (B, ph*pw, 4C)
//
// Output:
//   (B, C, 2ph, 2pw)
//
// Equivalent to:
//
//   latents = latents.reshape(B, ph, pw, C, 2, 2)
//   latents = latents.permute(0, 3, 1, 4, 2, 5)
//   latents = latents.reshape(B, C, 2ph, 2pw)
//
static Tensor unpack_latents(Tensor packed, int channels, int packed_h, int packed_w) {
    const int64_t B = packed.shape()[0];
    const int64_t N = packed.shape()[1];
    const int64_t C = channels;

    if (N != packed_h * packed_w)
        throw std::invalid_argument("unpack_latents(): invalid packed shape");

    auto t = packed.permute({0, 2, 1}).contiguous();
    t = t.reshape({B, 2 * C, 2, N});
    t = t.permute({0, 1, 3, 2}).contiguous();
    t = t.reshape({B, 2 * C, packed_h, 2 * packed_w});
    t = t.reshape({B * C, 2, packed_h, 2 * packed_w});
    t = t.permute({0, 2, 1, 3}).contiguous();
    t = t.reshape({B, C, 2 * packed_h, 2 * packed_w});

    return t;
}

// Graph-native 2x2 pack.
//
// Input:
//   latents: (B, C, 2ph, 2pw)
//
// Output:
//   (B, ph*pw, 4C)
//
// Equivalent to:
//
//   latents = latents.reshape(B, C, ph, 2, pw, 2)
//   latents = latents.permute(0, 2, 4, 1, 3, 5)
//   latents = latents.reshape(B, ph*pw, 4C)
//
static Tensor pack_latents(Tensor latents, int channels, int packed_h, int packed_w) {
    const int64_t B = latents.shape()[0];
    const int64_t C = channels;

    Tensor t = latents.reshape({B * C, 2 * packed_h, 2 * packed_w});
    t = t.reshape({B * C, packed_h, 2, 2 * packed_w});
    t = t.permute({0, 2, 1, 3}).contiguous();
    t = t.reshape({B, 2 * C, packed_h, 2 * packed_w});
    t = t.reshape({B, 2 * C, packed_h * packed_w, 2});
    t = t.permute({0, 1, 3, 2}).contiguous();
    t = t.reshape({B, 4 * C, packed_h * packed_w});
    return t.permute({0, 2, 1}).contiguous();
}

static std::vector<Image> latents_to_images(std::vector<float>&& data, int batch, int height, int width) {
    std::vector<Image> images;
    images.reserve(batch);

    const size_t w = (size_t)width;
    const size_t h = (size_t)height;
    const size_t plane = w * h;
    constexpr size_t channels = 3;

    for (int b = 0; b < batch; ++b) {
        std::vector<uint8_t> pixels(plane * channels);
        const float* src = data.data() + size_t(b) * channels * plane;

        for (size_t y = 0; y < h; ++y)
            for (size_t x = 0; x < w; ++x)
                for (size_t c = 0; c < channels; ++c) {
                    float v = src[c * plane + y * w + x];
                    v = std::clamp(v * 0.5f + 0.5f, 0.0f, 1.0f);
                    pixels[(y * w + x) * channels + c] = static_cast<uint8_t>(std::lrint(v * 255.0f));
                }

        images.emplace_back(w, h, channels, std::move(pixels));
    }
    return images;
}

static Tensor image_to_tensor(Runtime& runtime, const Image& img) {
    const size_t width = img.width();
    const size_t height = img.height();
    const size_t channels = img.channels();

    if (channels != 3) {
        throw std::invalid_argument(
            "image_to_tensor(): expected RGB image with 3 channels, got " +
            std::to_string(channels));
    }

    const auto& pixels = img.pixels();

    const size_t expected_size = width * height * channels;
    if (pixels.size() != expected_size) {
        throw std::invalid_argument(
            "image_to_tensor(): pixel buffer size does not match image dimensions");
    }

    return runtime.create<float>(
        {1, 3, static_cast<int64_t>(height), static_cast<int64_t>(width)},
        [pixels, width, height](std::mt19937&) {
            const size_t plane = width * height;

            std::vector<float> data(3 * plane);

            for (size_t y = 0; y < height; ++y) {
                for (size_t x = 0; x < width; ++x) {
                    const size_t src =
                        (y * width + x) * 3;

                    const size_t dst =
                        y * width + x;

                    // HWC uint8 -> CHW float32.
                    //
                    // Diffusers VAE expects image values in [-1, 1].
                    data[0 * plane + dst] =
                        static_cast<float>(pixels[src + 0]) / 127.5f - 1.0f;

                    data[1 * plane + dst] =
                        static_cast<float>(pixels[src + 1]) / 127.5f - 1.0f;

                    data[2 * plane + dst] =
                        static_cast<float>(pixels[src + 2]) / 127.5f - 1.0f;
                }
            }

            return data;
        });
}

std::vector<Image> Flux2KleinPipeline::operator ()(Scheduler& scheduler, const GenerationOptions& options) {
    if (options.height % vae_.scale_factor() != 0 ||
        options.width % vae_.scale_factor() != 0)
        throw std::runtime_error("height/width must be divisible by VAE's scale factor");

    auto seed = options.seed.value_or(std::random_device{}());

    auto latent_height = 2 * (options.height / (vae_.scale_factor() * 2));
    auto latent_width  = 2 * (options.width  / (vae_.scale_factor() * 2));

    const int packed_h = latent_height / 2;
    const int packed_w = latent_width / 2;
    const int batch = options.num_images_per_prompt;

    // 1. Generate text and image embeds
    std::vector<float> prompt_embeds_data, img_ids_data, txt_ids_data;
    Tensor::Shape prompt_embeds_shape, img_ids_shape, txt_ids_shape;
    {
        Runtime runtime(scheduler, seed);
        
        auto [embeds, txt_ids] = encode_prompt(runtime, batch, options.prompt, options.max_sequence_length);

        auto img_ids = prepare_img_ids(runtime, batch, packed_h, packed_w);

        Graph graph(runtime, {embeds, img_ids, txt_ids});
        Computation computation(graph);

        prompt_embeds_data = std::move(computation.read<float>(embeds));
        prompt_embeds_shape = embeds.shape();
        img_ids_data = std::move(computation.read<float>(img_ids));
        img_ids_shape = img_ids.shape();
        txt_ids_data = std::move(computation.read<float>(txt_ids));
        txt_ids_shape = txt_ids.shape();
    }

    // 2. Setup timesteps
    size_t num_ref_tokens = 0;
    {
        auto image_seq_len = int64_t(packed_h) * packed_w;
        
        // Handle img2img reference tokens
        if (!options.images.empty()) {
            num_ref_tokens = options.images.size() * packed_h * packed_w;
            image_seq_len += num_ref_tokens;
        }

        auto mu = compute_empirical_mu(image_seq_len, options.num_inference_steps);
        scheduler_.set_timesteps(options.num_inference_steps, mu);
    }
    
    // 3. Denoise in latent space
    std::vector<float> latents_data;
    Tensor::Shape latents_shape;
    {
        Runtime runtime(scheduler, seed);

        // Create graph inputs
        auto prompt_embeds = runtime.create<float>(prompt_embeds_shape, [&](std::mt19937&) { return std::move(prompt_embeds_data); });
        auto img_ids = runtime.create<float>(img_ids_shape, [&](std::mt19937&) { return std::move(img_ids_data); });
        auto txt_ids = runtime.create<float>(txt_ids_shape, [&](std::mt19937&) { return std::move(txt_ids_data); });
        auto latents = noisy_latents(runtime, batch, packed_h, packed_w, vae_.latent_channels());

        Tensor image_latents_concat;
        Tensor image_latent_ids_concat;

        if (!options.images.empty()) {
            std::vector<Tensor> packed_imgs;
            for (const auto& img : options.images) {
                auto img_tensor = image_to_tensor(runtime, img);
                auto dist = vae_.encode(runtime, img_tensor);
                
                // Pack (2x2 patchify + flatten)
                auto patched = pack_latents(dist.mode(), vae_.latent_channels(), packed_h, packed_w);
                
                // BN normalize
                auto bn_mean = vae_.bn().running_mean()->reshape({1, -1, 1, 1});
                auto bn_std = sqrt(vae_.bn().running_var()->reshape({1, -1, 1, 1}) + vae_.batch_norm_eps());
                patched = (patched - bn_mean) / bn_std;
                
                packed_imgs.push_back(patched);
            }

            image_latents_concat = Tensor::cat(
                packed_imgs,
                /*axis=*/1);

            image_latents_concat =
                repeat_batch(image_latents_concat, batch);

            image_latent_ids_concat =
                prepare_image_ids(
                    runtime,
                    batch,
                    options.images.size(),
                    packed_h,
                    packed_w);
        }

        auto latent_model_input = latents;
        auto latent_image_ids_input = img_ids;

        if (num_ref_tokens > 0) {
            latent_model_input = Tensor::cat({latents, image_latents_concat}, /*axis=*/1);
            latent_image_ids_input = Tensor::cat({img_ids, image_latent_ids_concat}, /*axis=*/1);
        }

        float current_timestep;
        auto timestep = runtime.create<float>({1}, [&](std::mt19937&) {
            return std::vector<float>({current_timestep});
        });

        timestep = timestep.expand(latents.shape()[0]); // Add batch
        timestep = timestep / 1000.0f;

        auto noise_pred = transformer_.forward(runtime,
            /*hidden_states=*/          latent_model_input,
            /*encoder_hidden_states=*/  prompt_embeds,
            /*timestep=*/               timestep,
            /*img_ids=*/                latent_image_ids_input,
            /*txt_ids=*/                txt_ids,
            /*guidance=*/               std::nullopt,
            /*num_ref_tokens=*/         num_ref_tokens
        );

        // Slice noise_pred to original latents shape if img2img
        if (num_ref_tokens > 0)
            noise_pred = noise_pred[{Tensor::Slice::all(), Tensor::Slice::range(0, latents.shape()[1])}];

        auto dt = runtime.create<float>({1}, [&](std::mt19937&) {
            return std::vector<float>({scheduler_.step()}); // We must step inside the computation
        });

        // Integrate latents over dt and assign latents <- latents+1
        auto next_latents = scheduler_.integrate(noise_pred, latents, dt);
        next_latents = next_latents.copy(latents);

        Graph denoise(runtime, {next_latents});
        ProgressBar progress("Denoising", scheduler_.get_timesteps().size());

        for (auto i = 0; i < scheduler_.get_timesteps().size(); ++i) {
            current_timestep = scheduler_.get_timesteps()[i];

            Computation computation(denoise);

            // Extract latents from the last denoising step
            if (i == scheduler_.get_timesteps().size() - 1) {
                latents_data = computation.read<float>(latents);
                latents_shape = latents.shape();
            }

            progress.update(i);
        }
    }

    // 4. Decode latents to pixels
    std::vector<Image> images;
    {
        Runtime runtime(scheduler, seed);

        auto latents = runtime.create<float>(latents_shape, [&](std::mt19937&) {
            return std::move(latents_data);
        });

        auto z = unpack_latents(latents, vae_.latent_channels(), packed_h, packed_w);

        // BN Unnormalize before VAE decode
        auto bn_mean = vae_.bn().running_mean()->reshape({1, -1, 1, 1});
        auto bn_std = sqrt(vae_.bn().running_var()->reshape({1, -1, 1, 1}) + vae_.batch_norm_eps());
        z = z * bn_std + bn_mean;

        auto decoded = vae_.decode(runtime, z);    // (B, 3, H, W)

        Graph graph(runtime, {decoded});
        Computation computation(graph);

        auto data = computation.read<float>(decoded);

        images = latents_to_images(std::move(data), batch, options.height, options.width);
    }

    return images;
}
