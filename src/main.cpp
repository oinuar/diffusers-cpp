#include "ggml/Backend.hpp"
#include "ggml/Scheduler.hpp"
#include "ggml/GGUFLoaderVisitor.hpp"
#include "nn/RethrowVisitor.hpp"
#include "diffusers/models/transformers/flux2/Flux2Transformer2DModel.hpp"
#include "diffusers/models/autoencoders/AutoencoderKLFlux2.hpp"
#include <iostream>
#include <filesystem>

int main() {
    ggml_time_init();
    ggml_log_set([](ggml_log_level, const char* text, void*) { std::cerr << text; }, nullptr);

    ggml_backend_load_all();

    Backend cpu(GGML_BACKEND_DEVICE_TYPE_CPU);

    std::filesystem::path path("../utils/convert-model/flux2-vae.gguf");

    AutoencoderKLFlux2 vae;

    GGUFLoaderVisitor loader(cpu, path);
    RethrowVisitor visitor(loader);
    vae.accept(visitor);
    visitor.rethrow();

    /*Scheduler scheduler({*cpu});

    Flux2Transformer2DModelCompute transformer;

    auto graph = scheduler.plan(transformer);
    transformer.compute(graph);*/

    return 0;
}
