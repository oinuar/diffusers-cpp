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
class Allocator;

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
        std::vector<Image> images = {};
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

    static Flux2KleinPipeline from_pretrained(
        Runtime& runtime,
        const std::filesystem::path& path,
        Allocator* vae_allocator = nullptr,
        Allocator* text_encoder_allocator = nullptr,
        Allocator* transformer_allocator = nullptr
    );

    // Latent shape conversions mirroring the static methods of the Python
    // Flux2KleinPipeline. Pure tensor ops with no model state.
    //
    //   pack_latents       (B, C, H, W)   -> (B, H*W, C)   _pack_latents
    //   unpack_latents     (B, H*W, C)    -> (B, C, H, W)  _unpack_latents_with_ids (canonical ids)
    //   patchify_latents   (B, C, 2H, 2W) -> (B, 4C, H, W) _patchify_latents
    //   unpatchify_latents (B, 4C, H, W)  -> (B, C, 2H, 2W) _unpatchify_latents
    static Tensor pack_latents(Tensor latents);
    static Tensor unpack_latents(Tensor packed, int packed_h, int packed_w);
    static Tensor patchify_latents(Tensor latents, int channels, int packed_h, int packed_w);
    static Tensor unpatchify_latents(Tensor latents, int channels, int packed_h, int packed_w);

    Flux2KleinPipeline(Flux2Transformer2DModel&& transformer,
                       AutoencoderKLFlux2&& vae,
                       Qwen3ForCausalLM&& text_encoder,
                       Qwen2TokenizerFast&& tokenizer);

    std::vector<Image> operator ()(Runtime& runtime, Allocator& allocator, GenerationOptions&& options);

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
        const std::string& prompt,
        size_t max_sequence_length,
        int batch,
        int packed_h,
        int packed_w,
        std::vector<Image>& images,
        Allocator* allocator = nullptr
    );

    Graph make_denoise_graph(
        Runtime& runtime,
        int batch,
        int packed_h,
        int packed_w,
        size_t num_ref_tokens,
        Tensor prompt_embeds,
        Tensor img_ids,
        Tensor txt_ids,
        Tensor latents,
        std::optional<Tensor> image_latents,
        std::optional<Tensor> image_latent_ids,
        float* timestep,
        float* dt
    );

    Graph make_decode_graph(
        Runtime& runtime,
        int packed_h,
        int packed_w,
        Tensor latents
    );

    const FlowMatchEulerDiscreteScheduler& scheduler() const {
        return scheduler_;
    }

private:
    std::tuple<Tensor, Tensor> encode_prompt(Runtime& runtime, Allocator* allocator, int batch, const std::string& prompt, size_t max_sequence_length);

    Flux2Transformer2DModel transformer_;
    AutoencoderKLFlux2 vae_;
    FlowMatchEulerDiscreteScheduler scheduler_;
    Qwen3ForCausalLM text_encoder_;
    Qwen2TokenizerFast tokenizer_;
};
