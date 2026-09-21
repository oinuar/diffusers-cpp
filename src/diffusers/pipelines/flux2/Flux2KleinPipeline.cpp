#include "diffusers/pipelines/flux2/Flux2KleinPipeline.hpp"
#include "nn/RethrowVisitor.hpp"
#include "nn/Parameter.hpp"
#include "nn/modules/BatchNorm2d.hpp"
#include "ggml/GGUFLoaderVisitor.hpp"
#include "ggml/Context.hpp"
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

static Tensor repeat_batch(const Tensor& t, int batch) {
    if (batch <= 1) return t;
    auto shape = t.shape();
    shape[0] = batch;
    return t.repeat(shape);
}

// 4D txt_ids: (B, L, 4) -> [0, 0, 0, l]
static Tensor prepare_txt_ids(Scope scope, int batch, int64_t seq_len) {
    // matches:
    //  t = torch.arange(1)
    //  h = torch.arange(1)
    //  w = torch.arange(1)
    //  l = torch.arange(L)
    //  coords = torch.cartesian_prod(t, h, w, l)
    // GGML RoPE expects position IDs to be 32b integers.
    return scope.context().create<int32_t>({batch, seq_len, 4},
        [=](std::mt19937&) {
            std::vector<int32_t> ids(size_t(batch) * seq_len * 4, 0);
            for (int b = 0; b < batch; ++b) {
                for (int l = 0; l < seq_len; ++l) {
                    int32_t* row = &ids[(size_t(b) * seq_len + l) * 4];
                    row[0] = 0; // T
                    row[1] = 0; // H
                    row[2] = 0; // W
                    row[3] = l; // L
                }
            }
            return std::move(ids);
        });
}

// 4D img_ids: (B, N, 4) -> [0, y, x, 0]
static Tensor prepare_img_ids(Scope scope, int batch, int packed_h, int packed_w) {
    int64_t N = int64_t(packed_h) * packed_w;

    // GGML RoPE expects position IDs to be 32b integers.
    return scope.context().create<int32_t>({batch, N, 4},
        [=](std::mt19937&) {
            std::vector<int32_t> ids(size_t(batch) * N * 4, 0);
            for (int b = 0; b < batch; ++b) {
                for (int y = 0; y < packed_h; ++y) {
                    for (int x = 0; x < packed_w; ++x) {
                        int32_t* row = &ids[(size_t(b) * N + y * packed_w + x) * 4];
                        row[0] = 0; // T
                        row[1] = y; // H
                        row[2] = x; // W
                        row[3] = 0; // L
                    }
                }
            }
            return std::move(ids);
        });
}

// 4D image_latent_ids for img2img: (B, N_ref, 4) -> [scale + i*scale, y, x, 0]
static Tensor prepare_ref_image_ids(Scope scope, int batch, const std::vector<Image>& images, int vae_multiple, size_t* num_ref_tokens) {
    *num_ref_tokens = 0;

    for (const auto& image : images) {
        const int packed_h = static_cast<int>(image.height() / vae_multiple);
        const int packed_w = static_cast<int>(image.width() / vae_multiple);

        *num_ref_tokens +=
            int64_t(packed_h) * packed_w;
    }

    int64_t total_tokens = *num_ref_tokens;

    // GGML RoPE expects position IDs to be 32b integers.
    return scope.context().create<int32_t>(
        {batch, total_tokens, 4},
        [=, &images](std::mt19937&) {
            std::vector<int32_t> ids(
                size_t(batch) * total_tokens * 4,
                0);

            for (int b = 0; b < batch; ++b) {
                int64_t offset = 0;

                for (size_t i = 0; i < images.size(); ++i) {
                    const auto& image = images[i];

                    const int packed_h = static_cast<int>(image.height() / vae_multiple);
                    const int packed_w = static_cast<int>(image.width() / vae_multiple);

                    const int32_t t =
                        10 +
                        10 * static_cast<int32_t>(i);

                    for (int y = 0; y < packed_h; ++y) {
                        for (int x = 0; x < packed_w; ++x) {
                            int32_t* row =
                                &ids[
                                    (size_t(b) * total_tokens +
                                     offset) * 4
                                ];

                            row[0] = t;
                            row[1] = y;
                            row[2] = x;
                            row[3] = 0;

                            ++offset;
                        }
                    }
                }
            }

            return ids;
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

static Tensor make_packed_latents(Scope scope, int batch, int packed_h, int packed_w, int num_latent_channels) {
    const int64_t token_dim = int64_t(num_latent_channels) * 4; // C * 4
    const size_t count = size_t(batch) * packed_h * packed_w * token_dim;

    return scope.context().create<float>({batch, int64_t(packed_h) * packed_w, token_dim},
        [=](std::mt19937& rng) {
            std::vector<float> noise(count);
            std::normal_distribution<float> normal(0.0f, 1.0f);
            for (float& v : noise) v = normal(rng);
            return std::move(noise);
        });
}

static Tensor make_init_latents(Scope scope, int batch, int packed_h, int packed_w, int num_latent_channels, std::vector<float>&& init_latents) {
    const int64_t token_dim = int64_t(num_latent_channels) * 4; // C * 4
    const size_t count = size_t(batch) * packed_h * packed_w * token_dim;

    return scope.context().create<float>({batch, int64_t(packed_h) * packed_w, token_dim},
        [init_latents = std::move(init_latents)](std::mt19937&) {
            return init_latents;
        });
}

static std::vector<Image> latents_to_images(const std::vector<float>& data, int batch, int height, int width) {
    std::vector<Image> images;
    images.reserve(batch);

    const size_t w = (size_t)width;
    const size_t h = (size_t)height;
    const size_t plane = w * h;
    constexpr size_t channels = 3;

    for (int b = 0; b < batch; ++b) {
        std::vector<uint8_t> pixels(plane * channels, 0);
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

    return std::move(images);
}

static Tensor image_to_tensor(Scope scope, const Image& img) {
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

    return scope.context().create<float>(
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
                    // Matches VaeImageProcessor.preprocess():
                    // (image / 255.0) * 2.0 - 1.0, mapping [0, 255] to [-1, 1].
                    data[0 * plane + dst] =
                        2.0f * (static_cast<float>(pixels[src + 0]) / 255.0f) - 1.0f;

                    data[1 * plane + dst] =
                        2.0f * (static_cast<float>(pixels[src + 1]) / 255.0f) - 1.0f;

                    data[2 * plane + dst] =
                        2.0f * (static_cast<float>(pixels[src + 2]) / 255.0f) - 1.0f;
                }
            }

            return data;
        });
}

static Image preprocess_reference_image(const Image& image, int multiple, double target_area = 1024.0 * 1024.0) {
    if (image.channels() != 3)
        throw std::invalid_argument(
            "preprocess_reference_image(): expected RGB image");

    const double width = static_cast<double>(image.width());
    const double height = static_cast<double>(image.height());
    const double area = width * height;

    // Flux2/Klein normalizes the input image to approximately
    // one megapixel while preserving the original aspect ratio.
    //
    // Do not upscale images that are already <= target area.
    const double scale = std::min(1.0, std::sqrt(target_area / area));

    auto target_width = static_cast<size_t>(std::round(width * scale));
    auto target_height = static_cast<size_t>(std::round(height * scale));

    // The VAE operates on dimensions that must be divisible by
    // its spatial scale factor, and Flux2 subsequently performs
    // 2x2 latent patchification.
    target_width = std::max<size_t>(multiple, (target_width / multiple) * multiple);
    target_height = std::max<size_t>(multiple, (target_height / multiple) * multiple);

    // resize_and_crop() preserves the aspect ratio and center-crops
    // to the exact dimensions required by the VAE.
    return image.resize_and_crop(target_width, target_height);
}

#if 0
Flux2KleinPipeline::Flux2KleinPipeline(
    Flux2Transformer2DModel&& transformer,
    AutoencoderKLFlux2&& vae,
    Qwen3ForCausalLM&& text_encoder,
    Qwen2TokenizerFast&& tokenizer
) : transformer_(std::move(transformer)),
    vae_(std::move(vae)),
    text_encoder_(std::move(text_encoder)),
    tokenizer_(std::move(tokenizer))
{
}

std::tuple<std::optional<Graph>, std::optional<Tensor>> Flux2KleinPipeline::make_vae_encode_graph(
    Scope scope,
    Scheduler& scheduler,
    const std::vector<Image>& images,
    int batch
) {
    auto vae_multiple = vae_.scale_factor() * 2;

    if (images.empty())
        return {std::nullopt, std::nullopt};

    std::vector<Tensor> packed_imgs;
    packed_imgs.reserve(images.size());

    // TODO: maybe encode only one image in this graph? it would better match decode then
    for (auto& img : images) {
        auto img_tensor = image_to_tensor(scope, img);
        auto dist = vae_.encode(scope, img_tensor);
        auto mode = dist.mode(); // Shape: (B, C, H, W)

        auto packed_h = img.height() / vae_multiple;
        auto packed_w = img.width() / vae_multiple;

        // 1. Patchify: (B, C, H, W) -> (B, 4C, H/2, W/2)
        auto patched_mode = patchify_latents(mode, vae_.latent_channels(), packed_h, packed_w);

        // 2. Apply BN normalization to patchified latents
        auto bn_mean = (*vae_.bn()->running_mean())->reshape({1, -1, 1, 1});
        auto bn_std = sqrt((*vae_.bn()->running_var())->reshape({1, -1, 1, 1}) + vae_.batch_norm_eps());
        patched_mode = (patched_mode - bn_mean) / bn_std; // Apply to (B, 4C, H/2, W/2)

        // 3. Pack latents: (B, 4C, H/2, W/2) -> (B, H/2 * W/2, 4C)
        auto packed = pack_latents(patched_mode);

        packed_imgs.push_back(packed);
    }

    // Concatenate reference images along the token dimension:
    //
    // (1, N1, 4C)
    // (1, N2, 4C)
    //       ↓
    // (1, N1 + N2, 4C)
    auto image_latents = Tensor::cat(packed_imgs, /*axis=*/1);

    // Repeat reference latents for num_images_per_prompt.
    image_latents = repeat_batch(image_latents, batch);

    return {
        Graph(scheduler, scope.context(), {image_latents}),
        image_latents
    };
}

std::tuple<Graph, Tensor> Flux2KleinPipeline::make_text_encoder_graph(
    Scope scope,
    Scheduler& scheduler,
    int batch,
    const std::string& prompt,
    size_t max_sequence_length
) {
    // Apply chat template to prompt
    std::string text = tokenizer_.apply_chat_template(
        {{"user", prompt}},
        /*add_generation_prompt=*/true,
        /*enable_thinking=*/false
    );

    std::vector<int> mask;
    size_t num_real_tokens;
    auto tokens = tokenizer_.encode(text, max_sequence_length, &mask, &num_real_tokens);

    auto input_ids = scope.context().create<int32_t>({batch, (int64_t)tokens.size()},
        [=](std::mt19937&) {
            std::vector<int32_t> ids(size_t(batch) * tokens.size());
            for (auto b=0; b<batch; ++b) std::copy(tokens.begin(), tokens.end(), ids.begin() + b * tokens.size());
            return std::move(ids);
        });

    auto attention_mask = scope.context().create<float>({batch, (int64_t)mask.size()},
        [=](std::mt19937&) {
            std::vector<float> m(size_t(batch) * mask.size());
            for (auto b=0; b<batch; ++b) std::copy(mask.begin(), mask.end(), m.begin() + b * mask.size());
            return std::move(m);
        });

    std::vector<Tensor> hidden_states;

    text_encoder_.forward(
        scope,
        input_ids,
        attention_mask,
        std::nullopt, // position_ids
        std::nullopt, // past_key_values
        std::nullopt, // inputs_embeds
        std::nullopt, // labels
        false, // use_cache
        0, // logits_to_keep
        &hidden_states
    );

    auto l9 = hidden_states[9];
    auto l18 = hidden_states[18];
    auto l27 = hidden_states[27];

    if (!l9 || !l18 || !l27)
        throw std::runtime_error("Hidden state extraction failed");

    // Stack along a new dimension (axis=1) -> (B, 3, seq, hidden)
    auto prompt_embeds = Tensor::stack({l9, l18, l27}, /*axis=*/1);
    
    // Permute to (B, seq, 3, hidden)
    prompt_embeds = prompt_embeds.permute({0, 2, 1, 3});
    
    // Reshape to (B, seq, 3 * hidden)
    int64_t hidden_dim = l9.shape()[2]; // Assuming (B, seq, hidden)
    prompt_embeds = prompt_embeds.reshape({batch, (int64_t)tokens.size(), 3 * hidden_dim});

    return {
        Graph(scheduler, scope.context(), {prompt_embeds}),
        prompt_embeds
    };
}

Graph Flux2KleinPipeline::make_denoise_graph(
    Scope scope,
    Scheduler& scheduler,
    int batch,
    int packed_h,
    int packed_w,
    size_t max_sequence_length,
    Tensor latents,
    Tensor prompt_embeds,
    std::optional<Tensor> image_latents,
    const std::vector<Image>& images,
    float* timestep,
    float* dt
) {
    auto img_ids = prepare_img_ids(scope, batch, packed_h, packed_w);
    auto txt_ids = prepare_txt_ids(scope, batch, max_sequence_length);

    auto latent_model_input = latents;
    auto latent_image_ids = img_ids;
    size_t num_ref_tokens = 0;

    if (!images.empty()) {
        latent_model_input = Tensor::cat({latents, *image_latents}, /*axis=*/1);

        auto image_latent_ids = prepare_ref_image_ids(scope, batch, images, vae_.scale_factor() * 2, &num_ref_tokens);
        latent_image_ids = Tensor::cat({img_ids, image_latent_ids}, /*axis=*/1);
    }

    auto timestep_tensor = scope.context().value<float>({batch}, [batch, timestep](std::mt19937&) {
        return std::vector<float>(batch, *timestep);
    });

    auto noise_pred = transformer_.forward(scope,
        /*hidden_states=*/          latent_model_input,
        /*encoder_hidden_states=*/  prompt_embeds,
        /*timestep=*/               timestep_tensor / 1000.0f,
        /*img_ids=*/                img_ids,
        /*txt_ids=*/                txt_ids,
        /*guidance=*/               std::nullopt,
        /*num_ref_tokens=*/         num_ref_tokens
    );

    // Slice noise_pred to original latents shape if img2img
    if (num_ref_tokens > 0)
        noise_pred = noise_pred[{Tensor::Slice::all(), Tensor::Slice::range(0, latents.shape()[1])}];

    auto dt_tensor = scope.context().value<float>({}, [dt](std::mt19937&) {
        return std::vector<float>{*dt};
    });

    // Integrate latents over dt
    auto next_latents = scheduler_.integrate(scope, noise_pred, latents, dt_tensor);

    return Graph(scheduler, scope.context(), {next_latents});
}

Graph Flux2KleinPipeline::make_vae_decode_graph(
    Scope scope,
    Scheduler& scheduler,
    int packed_h,
    int packed_w,
    Tensor latents
) {
    // 1. Unpack to patch grid (B, 4C, ph, pw)
    auto z_packed = unpack_latents(latents, packed_h, packed_w);

    // 2. BN Unnormalize before VAE decode (Broadcasts over the 4C channels)
    auto bn_mean = (*vae_.bn()->running_mean())->reshape({1, -1, 1, 1});
    auto bn_std = sqrt((*vae_.bn()->running_var())->reshape({1, -1, 1, 1}) + vae_.batch_norm_eps());
    z_packed = z_packed * bn_std + bn_mean;

    // 3. Unpatchify to spatial latents (B, C, 2ph, 2pw)
    auto z = unpatchify_latents(z_packed, vae_.latent_channels(), packed_h, packed_w);

    // 4. VAE decode
    auto decoded = vae_.decode(scope, z);    // (B, 3, W, H)

    return Graph(scheduler, scope.context(), {decoded});
}

std::vector<Image> Flux2KleinPipeline::operator ()(
    Allocator& allocator,
    Scheduler& scheduler,
    Context& vae_context,
    Context& text_encoder_context,
    Context& transformer_context,
    GenerationOptions&& options
) {
    if (options.height % vae_.scale_factor() != 0 ||
        options.width % vae_.scale_factor() != 0)
        throw std::runtime_error("height/width must be divisible by VAE's scale factor");

    auto vae_multiple = vae_.scale_factor() * 2;

    auto target_height = (options.height / vae_multiple) * vae_multiple;
    auto target_width = (options.width / vae_multiple) * vae_multiple;

    auto latent_height = 2 * (target_height / vae_multiple);
    auto latent_width = 2 * (target_width / vae_multiple);

    auto packed_h = latent_height / 2;
    auto packed_w = latent_width / 2;

    auto batch = options.num_images_per_prompt;

    auto image_seq_len = static_cast<int64_t>(packed_h) * packed_w;

    ProgressBar progress("Generating", 1);

    // The temporary (computational) tensors of each graph live in its own
    // context and are allocated by the scheduler; only the state created in
    // state_context crosses graph boundaries, and the allocator keeps it
    // alive for the whole generation.
    Context vae_encode_context;
    Context text_encoder_graph_context;
    Context denoise_context;
    Context decode_context;

    // 1. Preprocess reference images.
    for (auto &img : options.images)
        img = preprocess_reference_image(img, vae_multiple);

    // 2. Encode images to latents.
    auto [vae_encode_graph, image_latents] = make_vae_encode_graph(
        vae_encode_context,
        scheduler,
        options.images,
        batch);

    // 3. Encode the prompt into text embeddings.
    auto [text_encoder_graph, prompt_embeds] = make_text_encoder_graph(
        text_encoder_graph_context,
        scheduler,
        batch,
        options.prompt,
        options.max_sequence_length);

    // 4. Initial latents.
    auto latents = options.init_latents
        ? make_init_latents(denoise_context, batch, packed_h, packed_w, vae_.latent_channels(), std::move(*options.init_latents))
        : make_packed_latents(denoise_context, batch, packed_h, packed_w, vae_.latent_channels());

    // 5. One denoising step per computation.
    float timestep, dt;
    auto denoise_graph = make_denoise_graph(
        denoise_context,
        scheduler,
        batch,
        packed_h,
        packed_w,
        options.max_sequence_length,
        latents,
        prompt_embeds,
        image_latents,
        options.images,
        &timestep,
        &dt);

    // 6. Decode latents to pixels.
    auto decode_graph = make_vae_decode_graph(
        decode_context,
        scheduler,
        packed_h,
        packed_w,
        latents);

    // TODO: allocate everything
    //   allocator.allocate({state_context, vae_context, text_encoder_context, transformer_context});

    // 7. VAE encode
    if (vae_encode_graph) {
        progress.push("VAE encoding", 1);

        Computation computation(allocator, *vae_encode_graph, {&vae_context}, &progress);
        computation();

        progress.next();
        progress.pop();
    }

    // 8. Text encode
    {
        progress.push("Text encoding", 1);

        Computation computation(allocator, text_encoder_graph, {&text_encoder_context}, &progress);
        computation();

        progress.next();
        progress.pop();
    }

    // 9. Denoise in latent space
    {
        // Scheduler shifting is based only on the generated image
        // sequence length, not reference-image tokens.
        auto mu = compute_empirical_mu(image_seq_len, options.num_inference_steps);
        const int n = options.num_inference_steps;

        std::vector<float> sigmas(n);
        for (int i = 0; i < n; ++i)
            sigmas[i] =
                static_cast<float>(1.0 - static_cast<double>(i) / n);

        auto schedule = std::move(scheduler_.schedule(n, mu, std::move(sigmas)));

        progress.push("Denoising", schedule.size());

        Computation computation(allocator, denoise_graph, {&transformer_context}, &progress);

        for (const auto& step : schedule) {
            timestep = step.timestep;
            dt = step.dt;

            computation();

            progress.next();
        }

        progress.pop();
    }

    // 10. Decode latents to pixels
    std::vector<Image> images;
    {
        progress.push("VAE decoding", 1);

        Computation computation(allocator, decode_graph, {&vae_context}, &progress);

        auto decoded = computation().results().at(0);
        auto data = decode_graph.context().read<float>(decoded);

        images = std::move(latents_to_images(data, batch, target_height, target_width));

        progress.next();
        progress.pop();
    }

    progress.next();

    return std::move(images);
}

Flux2KleinPipeline Flux2KleinPipeline::from_pretrained(Context& vae_context, Context& text_encoder_context, Context& transformer_context, const std::filesystem::path& path) {
    // 1. Initialize models from config
    Qwen3ForCausalLM text_encoder(Qwen3Config::from_file(path / "text_encoder" / "config.json"));
    Flux2Transformer2DModel transformer(Flux2Transformer2DModel::Config::from_file(path / "transformer" / "config.json"));
    AutoencoderKLFlux2 vae(AutoencoderKLFlux2::Config::from_file(path / "vae" / "config.json"));

    // 2. Load VAE
    {
        GGUFLoaderVisitor vae_loader(vae_context, path / "vae");
        RethrowVisitor vae_visitor(vae_loader);
        vae.accept(vae_visitor);
        vae_visitor.rethrow();
        vae_loader.validate();
    }

    // 3. Load text encoder
    {
        GGUFLoaderVisitor text_encoder_loader(text_encoder_context, path / "text_encoder");
        RethrowVisitor text_encoder_visitor(text_encoder_loader);
        text_encoder.accept(text_encoder_visitor);
        text_encoder_visitor.rethrow();
        text_encoder_loader.validate();
    }

    // 4. Load transformer
    {
        GGUFLoaderVisitor transformer_loader(transformer_context, path / "transformer");
        RethrowVisitor transformer_visitor(transformer_loader);
        transformer.accept(transformer_visitor);
        transformer_visitor.rethrow();
        transformer_loader.validate();
    };

    // 5. Load tokenizer
    auto tokenizer = Qwen2TokenizerFast::from_pretrained(path / "tokenizer");

    // 6. Construct the pipeline
    Flux2KleinPipeline pipeline(
        std::move(transformer),
        std::move(vae),
        std::move(text_encoder),
        std::move(tokenizer)
    );

    return std::move(pipeline);
}
#endif

// 1. Unpacks from (B, ph*pw, 4C) to (B, 4C, ph, pw)
Tensor Flux2KleinPipeline::unpack_latents(Tensor packed, int packed_h, int packed_w) {
    const int64_t B = packed.shape()[0];
    const int64_t C4 = packed.shape()[2];
    auto t = packed.permute({0, 2, 1}).contiguous();
    return t.reshape({B, C4, packed_h, packed_w});
}

// 2. Unpatchifies from (B, 4C, ph, pw) to (B, C, 2ph, 2pw)
// Uses a 4D interleaving trick to avoid 5D/6D tensors
Tensor Flux2KleinPipeline::unpatchify_latents(Tensor patched, int channels, int packed_h, int packed_w) {
    const int64_t B = patched.shape()[0];
    const int64_t C = channels;
    const int64_t N = packed_h * packed_w;

    auto t = patched.reshape({B, 4 * C, N});
    t = t.reshape({B, 2 * C, 2, N});
    t = t.permute({0, 1, 3, 2}).contiguous();
    t = t.reshape({B, 2 * C, packed_h, 2 * packed_w});
    t = t.reshape({B * C, 2, packed_h, 2 * packed_w});
    t = t.permute({0, 2, 1, 3}).contiguous();
    t = t.reshape({B, C, 2 * packed_h, 2 * packed_w});

    return t;
}

// 3. Packs from (B, 4C, ph, pw) to (B, ph*pw, 4C)
Tensor Flux2KleinPipeline::pack_latents(Tensor latents) {
    const int64_t B = latents.shape()[0];
    const int64_t C4 = latents.shape()[1];
    const int64_t ph = latents.shape()[2];
    const int64_t pw = latents.shape()[3];
    const int64_t N = ph * pw;

    auto t = latents.reshape({B, C4, N});
    return t.permute({0, 2, 1}).contiguous();
}

// 4. Patchifies from (B, C, 2ph, 2pw) to (B, 4C, ph, pw)
// Inverse of the 4D interleaving trick
Tensor Flux2KleinPipeline::patchify_latents(Tensor latents, int channels, int packed_h, int packed_w) {
    const int64_t B = latents.shape()[0];
    const int64_t C = channels;
    const int64_t N = packed_h * packed_w;

    auto t = latents.reshape({B * C, packed_h, 2, 2 * packed_w});
    t = t.permute({0, 2, 1, 3}).contiguous();
    t = t.reshape({B, 2 * C, packed_h, 2 * packed_w});
    t = t.reshape({B, 2 * C, N, 2});
    t = t.permute({0, 1, 3, 2}).contiguous();
    t = t.reshape({B, 4 * C, N});
    t = t.reshape({B, 4 * C, packed_h, packed_w});

    return t;
}
