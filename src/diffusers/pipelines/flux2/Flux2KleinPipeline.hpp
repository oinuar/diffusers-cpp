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

    // ---------------------------------------------------------------------
    // Individual computation stages.
    //
    // Each stage takes the caller's Scope directly and builds its tensors
    // in the caller's scope context. They are deliberately decoupled from
    // the pipeline's own computation chain (operator()), so tests can invoke
    // them inside the test runner's own computation scope.
    // ---------------------------------------------------------------------

    // Encodes reference images into packed latents (B, N, 4 * C), where N
    // is the sum of the packed token counts of all images. Returns
    // std::nullopt when there are no images.
    std::optional<Tensor> encode_images(Scope scope, const std::vector<Image>& images, int batch);

    // Encodes the prompt into text embeddings (B, L, 3 * hidden_dim).
    Tensor encode_prompt(Scope scope, int batch, const std::string& prompt, size_t max_sequence_length);

    // Runs one denoising step: the transformer forward pass followed by the
    // scheduler integration. When image_latents is provided, the reference
    // tokens (image_latents + image_latent_ids) are appended to the latents
    // and noise_pred is sliced back to the latent length.
    Tensor denoise_step(
        Scope scope,
        const Tensor& latents,
        const Tensor& prompt_embeds,
        const Tensor& img_ids,
        const Tensor& txt_ids,
        const std::optional<Tensor>& image_latents,
        const std::optional<Tensor>& image_latent_ids,
        const Tensor& timestep,
        const Tensor& dt);

    // Decodes packed latents into pixels (B, 3, H, W).
    Tensor decode(Scope scope, const Tensor& latents, int packed_h, int packed_w);

    // 4D txt_ids: (B, L, 4) -> [0, 0, 0, l]
    static Tensor prepare_txt_ids(Scope scope, int batch, int64_t seq_len);

    // 4D img_ids: (B, N, 4) -> [0, y, x, 0]
    static Tensor prepare_img_ids(Scope scope, int batch, int packed_h, int packed_w);

    // Converts the raw decoded values (B, 3, H, W) read back from the
    // execution result into RGB images in [0, 255]. Computation<Image> is
    // not supported, so this conversion is performed on the CPU side.
    static std::vector<Image> to_images(const std::vector<float>& data, int batch, int height, int width);
    // Normalizes a reference image the same way the Python pipeline's
    // __call__ prepares condition images: resizes images above the target
    // area to the target area, then crops to the nearest multiple of the VAE
    // spatial multiple (vae_scale_factor * 2).
    static Image preprocess_reference_image(const Image& image, int multiple, double target_area = 1024.0 * 1024.0);
    int64_t vae_scale_factor() const {
        return vae_.scale_factor();
    }

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
