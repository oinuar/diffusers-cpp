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
class Scheduler;

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

    std::vector<Image> operator ()(Scheduler& scheduler, const GenerationOptions& options);

private:
    std::pair<Tensor, Tensor> encode_prompt(Runtime& runtime, int batch, const std::string& prompt, size_t max_sequence_length);

    Flux2Transformer2DModel transformer_;
    AutoencoderKLFlux2 vae_;
    FlowMatchEulerDiscreteScheduler scheduler_;
    Qwen3ForCausalLM text_encoder_;
    Qwen2TokenizerFast tokenizer_;
};
