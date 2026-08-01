#include "ggml/Backend.hpp"
#include "ggml/Scheduler.hpp"
#include "ggml/GGUFLoaderVisitor.hpp"
#include "nn/RethrowVisitor.hpp"
#include "diffusers/models/transformers/flux2/Flux2Transformer2DModel.hpp"
#include "diffusers/models/autoencoders/AutoencoderKLFlux2.hpp"
#include "transformers/models/qwen3/Qwen3Config.hpp"
#include "transformers/models/qwen3/Qwen3ForCausalLM.hpp"
#include <iostream>
#include <filesystem>

int main() {
    ggml_time_init();
    ggml_log_set([](ggml_log_level, const char* text, void*) { std::cerr << text; }, nullptr);

    ggml_backend_load_all();

    Backend cpu(GGML_BACKEND_DEVICE_TYPE_CPU);

    /*std::filesystem::path path("../utils/convert-model/flux2-vae.gguf");

    AutoencoderKLFlux2 vae;

    GGUFLoaderVisitor loader(cpu, path);
    RethrowVisitor visitor(loader);
    vae.accept(visitor);
    visitor.rethrow();*/

    Qwen3ForCausalLM::from_pretrained(cpu, Qwen3Config::from_file("../utils/convert-model/text-encoder/config.json"), "../utils/convert-model/qwen3.gguf");

    return 0;
}
