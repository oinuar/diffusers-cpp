#include "ggml/Backend.hpp"
#include "ggml/Scheduler.hpp"
#include "models/diffusers/transformers/flux2/Flux2Transformer2DModel.hpp"
#include <iostream>

int main() {
    ggml_time_init();
    ggml_log_set([](ggml_log_level, const char* text, void*) { std::cerr << text; }, nullptr);

    ggml_backend_load_all();

    Backend cpu(GGML_BACKEND_DEVICE_TYPE_CPU);

    std::string path = "../examples/flux2-cli/utils/convert-model/flux2-klein-9b_q8_0.gguf";

    auto transformer = Flux2Transformer2DModel::from_pretrained(cpu, path);

    /*Scheduler scheduler({*cpu});

    Flux2Transformer2DModelCompute transformer;

    auto graph = scheduler.plan(transformer);
    transformer.compute(graph);*/

    return 0;
}
