#pragma once

#include "ggml/Tensor.hpp"
#include "ggml/Graph.hpp"
#include "Image.hpp"
#include "transformers/models/qwen3/Qwen3ForCausalLM.hpp"
#include "transformers/models/qwen2/Qwen2TokenizerFast.hpp"
#include "diffusers/models/autoencoders/AutoencoderKLFlux2.hpp"
#include "diffusers/models/transformers/flux2/Flux2Transformer2DModel.hpp"
#include "diffusers/schedulers/FlowMatchEulerDiscreteScheduler.hpp"
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include <filesystem>

class Backend;
class Runtime;
class Scheduler;

// 
//                     ┌────────────────────┐
// prompt ────────────►│ Qwen3 text encoder │
//                     └─────────┬──────────┘
//                               │
//                     prompt_embeds + txt_ids
//                               │
//                               ▼
//                          ┌─────────┐
// noise ────────────────►  │         │
// timestep ─────────────►  │ Flux2   │
// img_ids ──────────────►  │transform│
// image latents ────────►  └─────────┘
//                               │
//                               ▼
//                          noise_pred
//                               │
//                               ▼
//                          scheduler
//                               │
//                               ▼
//                          next_latents
//                               │
//                          repeat N times
//                               │
//                               ▼
//                          unpack / BN
//                               │
//                               ▼
//                             VAE
//                               │
//                               ▼
//                            image
class Flux2KleinPipeline {
public:
    struct GenerationOptions {
        std::string prompt;
        std::vector<Image> images;
        int height = 1024;
        int width = 1024;
        int num_inference_steps = 4;      // Klein default (distilled)
        float guidance_scale = 0.0f;      // unused: Klein is guidance-distilled
        int num_images_per_prompt = 1;
        std::optional<uint64_t> seed = std::nullopt;
        // Optional initial latents in the packed representation
        // (batch, packed_h * packed_w, 4 * latent_channels).
        // If nullopt, random noise is generated instead.
        std::optional<std::vector<float>> init_latents;
        size_t max_sequence_length = 512;
    };

    Flux2KleinPipeline from_pretrained(Backend& loader_backend, const std::filesystem::path& path);

    Flux2KleinPipeline(Flux2Transformer2DModel&& transformer,
                       AutoencoderKLFlux2&& vae,
                       Qwen3ForCausalLM&& text_encoder,
                       Qwen2TokenizerFast&& tokenizer)
        : transformer_(std::move(transformer)), vae_(std::move(vae)),
          scheduler_(), text_encoder_(std::move(text_encoder)),
          tokenizer_(std::move(tokenizer)) {}

    std::vector<Image> operator ()(Runtime& runtime, GenerationOptions&& options);

    struct Embeddings {
        Graph graph;
        Tensor prompt_embeds;
        Tensor txt_ids;
        Tensor img_ids;
        std::optional<Tensor> image_latents_concat;
        std::optional<Tensor> image_latent_ids_concat;
    };

    Embeddings make_embeddings_graph(
        Runtime& runtime,
        int batch,
        const std::string& prompt,
        size_t max_sequence_length,
        int packed_h,
        int packed_w,
        std::vector<Image>& images);

    struct Denoise {
        Graph graph;
        Tensor latents;
    };

    Denoise make_denoise_graph(
        Runtime& runtime,
        int batch,
        int packed_h,
        int packed_w,
        size_t num_ref_tokens,
        Tensor::Shape prompt_embeds_shape,
        Tensor::Shape img_ids_shape,
        Tensor::Shape txt_ids_shape,
        Tensor::Shape image_latents_shape,
        Tensor::Shape image_latent_ids_shape,
        std::vector<float>* packed_latents,
        std::vector<float>* prompt_embeds_data,
        std::vector<float>* img_ids_data,
        std::vector<float>* txt_ids_data,
        std::vector<float>* image_latents_data,
        std::vector<float>* image_latent_ids_data,
        float* timestep,
        float* dt
    );

    Graph make_decode_graph(
        Runtime& runtime,
        int packed_h,
        int packed_w,
        Tensor::Shape latents_shape,
        std::vector<float>* latents_data
    );

    const FlowMatchEulerDiscreteScheduler& scheduler() const {
        return scheduler_;
    }

private:
    std::tuple<Tensor, Tensor> encode_prompt(Runtime& runtime, int batch, const std::string& prompt, size_t max_sequence_length);

    Flux2Transformer2DModel transformer_;
    AutoencoderKLFlux2 vae_;
    FlowMatchEulerDiscreteScheduler scheduler_;
    Qwen3ForCausalLM text_encoder_;
    Qwen2TokenizerFast tokenizer_;
};
