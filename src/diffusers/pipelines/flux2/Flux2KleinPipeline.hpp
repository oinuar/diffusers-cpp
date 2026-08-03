#pragma once

#include "ggml/Tensor.hpp"
#include "Image.hpp"
#include "transformers/models/qwen3/Qwen3ForCausalLM.hpp"
#include "transformers/models/qwen2/Qwen2TokenizerFast.hpp"
#include "diffusers/models/autoencoders/AutoencoderKLFlux2.hpp"
#include "diffusers/models/transformers/flux2/Flux2Transformer2DModel.hpp"
#include "diffusers/schedulers/FlowMatchEulerDiscreteScheduler.hpp"
#include <cstdint>
#include <string>
#include <vector>
#include <filesystem>

class Backend;
class Runtime;

class Flux2KleinPipeline {
public:
    struct GenerationOptions {
        int height = 1024;
        int width = 1024;
        int num_inference_steps = 4;      // Klein default (distilled)
        float guidance_scale = 0.0f;      // unused: Klein is guidance-distilled
        int num_images_per_prompt = 1;
        uint64_t seed = 0;                // 0 -> std::random_device
        int max_sequence_length = 512;
    };

    struct PipelineState {
        Tensor prompt_embeds;          // (hidden, seq, B)
        Tensor pooled_prompt_embeds;   // (hidden, 1, B)
        Tensor latents;                // (token_dim, N_img, B) packed
        Tensor img_ids;                // (3, N_img)
        Tensor txt_ids;                // (3, seq)
        std::vector<float> timesteps;
        std::vector<float> sigmas;
        int latent_height = 0;         // VAE latent pixels (= 2 * packed grid h)
        int latent_width = 0;
    };
    
    Flux2KleinPipeline from_pretrained(Backend& loader_backend, const std::filesystem::path& path);

    Flux2KleinPipeline(Flux2Transformer2DModel&& transformer,
                       AutoencoderKLFlux2&& vae,
                       Qwen3ForCausalLM&& text_encoder,
                       Qwen2TokenizerFast&& tokenizer)
        : transformer_(std::move(transformer)), vae_(std::move(vae)),
          scheduler_(), text_encoder_(std::move(text_encoder)),
          tokenizer_(std::move(tokenizer)) {}

    std::vector<Image> generate(Runtime& runtime, 
                                const std::string& prompt,
                                const GenerationOptions& options);

private:
    PipelineState prepare(Runtime& runtime, const std::string& prompt, const GenerationOptions& options);

    std::pair<Tensor, Tensor> encode_prompt(Runtime& runtime, 
                                            const std::string& prompt,
                                            const GenerationOptions& options);

    Tensor prepare_latents(Runtime& runtime, int batch, int packed_h, int packed_w);

    Tensor prepare_img_ids(Runtime& runtime, int packed_h, int packed_w);

    Tensor prepare_txt_ids(Runtime& runtime, int seq_len);

    void prepare_timesteps(PipelineState& state, const GenerationOptions& options);

    Tensor repeat_batch(const Tensor& t, int batch);

    Tensor build_step(Runtime& rt, const PipelineState& state, float timestep);

    Tensor build_decode(Runtime& rt, const PipelineState& state,
                        const GenerationOptions& options);

    std::vector<Image> latents_to_images(const Tensor& decoded, int batch,
                                         int height, int width);

    Flux2Transformer2DModel transformer_;
    AutoencoderKLFlux2 vae_;
    FlowMatchEulerDiscreteScheduler scheduler_;
    Qwen3ForCausalLM text_encoder_;
    Qwen2TokenizerFast tokenizer_;
};
