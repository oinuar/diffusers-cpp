#pragma once

#include "ggml/Tensor.hpp"
#include "Image.hpp"
#include "ggml/Computation.hpp"
#include "transformers/models/qwen3/Qwen3ForCausalLM.hpp"
#include "transformers/models/qwen2/Qwen2TokenizerFast.hpp"
#include "diffusers/models/autoencoders/AutoencoderKLFlux2.hpp"
#include "diffusers/models/transformers/flux2/Flux2Transformer2DModel.hpp"
#include "diffusers/schedulers/FlowMatchEulerDiscreteScheduler.hpp"
#include <cstdint>
#include <utility>
#include <optional>
#include <string>
#include <vector>
#include <filesystem>

class Backend;
class Context;
class Scheduler;
class Allocator;

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

    static Flux2KleinPipeline from_pretrained(Context& vae_context, Context& text_encoder_context, Context& transformer_context, const std::filesystem::path& path);

    Flux2KleinPipeline(Flux2Transformer2DModel&& transformer,
                       AutoencoderKLFlux2&& vae,
                       Qwen3ForCausalLM&& text_encoder,
                       Qwen2TokenizerFast&& tokenizer);
    
    Computation<Tensor> operator()(
        Context& vae_context,
        Context& text_encoder_context,
        Context& transformer_context,
        GenerationOptions&& options);

    #if 0
    
    std::vector<Image> operator ()(
        Allocator& allocator,
        Scheduler& scheduler,
        Context& vae_context,
        Context& text_encoder_context,
        Context& transformer_context,
        GenerationOptions&& options);

    //
    // Graph construction.
    //
    // Each graph is built on its own graph context: the temporary
    // (computational) tensors are allocated by the scheduler when the graph
    // is computed and may be reclaimed by subsequent graphs. Only the state
    // that crosses graph boundaries is created in the state context, where
    // the allocator keeps it alive for the whole generation. Computed
    // outputs that must cross a boundary are copied into the returned state
    // tensors right after the graph runs (see operator()).
    //
    struct VaeEncodeGraph {
        Graph graph;
        // State tensors shared with the denoise graph.
        Tensor image_latents;      // (B, N_ref, 4 * latent_channels)
        Tensor image_latent_ids;   // (B, N_ref, 4)
    };

    struct TextEncoderGraph {
        Graph graph;
        // State tensors shared with the denoise graph.
        Tensor prompt_embeds;      // (B, seq, 3 * hidden)
        Tensor txt_ids;            // (B, seq, 4)
    };

    // Encodes the reference images into packed, normalized latents
    // (img2img only). image_latents is computed into the graph context and
    // the returned state tensor holds the copy the denoise graph reads;
    // image_latent_ids is created directly in the state context.
    std::tuple<std::optional<Graph>, std::optional<Tensor>> make_vae_encode_graph(
        Scope scope,
        Scheduler& scheduler,
        const std::vector<Image>& images,
        int batch
    );

    // Encodes the prompt into text embeddings and creates the position ids.
    // prompt_embeds is computed into the graph context and the returned
    // state tensor holds the copy the denoise graph reads; txt_ids and
    // img_ids are created directly in the state context.
    std::tuple<Graph, Tensor> make_text_encoder_graph(
        Scope scope,
        Scheduler& scheduler,
        int batch,
        const std::string& prompt,
        size_t max_sequence_length
    );

    // One denoising step: transformer forward + scheduler integration.
    // The embeddings, ids and latents are read from the state context;
    // the next latents are computed into the graph context and copied back
    // into latents by the caller after each run.
    Graph make_denoise_graph(
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
    );

    // Unpacks, unnormalizes and unpatchifies the latents and runs the VAE
    // decoder. The latents are read from the state context; the decoded
    // image is computed into the graph context.
    Graph make_vae_decode_graph(
        Scope scope,
        Scheduler& scheduler,
        int packed_h,
        int packed_w,
        Tensor latents
    );
#endif
    const FlowMatchEulerDiscreteScheduler& scheduler() const {
        return scheduler_;
    }

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

private:
    Flux2Transformer2DModel transformer_;
    AutoencoderKLFlux2 vae_;
    FlowMatchEulerDiscreteScheduler scheduler_;
    Qwen3ForCausalLM text_encoder_;
    Qwen2TokenizerFast tokenizer_;
};
