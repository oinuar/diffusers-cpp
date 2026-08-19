#include "ggml/Backend.hpp"
#include "ggml/Runtime.hpp"
#include "ggml/Scheduler.hpp"
#include "ggml/Computation.hpp"
#include "ggml/GGUFLoaderVisitor.hpp"
#include "nn/RethrowVisitor.hpp"
#include "diffusers/pipelines/flux2/Flux2KleinPipeline.hpp"
#include <iostream>
#include <filesystem>

int main() {
    ggml_time_init();
    ggml_log_set([](ggml_log_level, const char* text, void*) { std::cerr << text; }, nullptr);

    ggml_backend_load_all();

    Backend cpu(GGML_BACKEND_DEVICE_TYPE_CPU);
    Backend gpu(GGML_BACKEND_DEVICE_TYPE_GPU);
    Scheduler scheduler({*gpu, *cpu}, 65536);
    Context context(65536);
    Runtime runtime(scheduler, context);

    auto pipeline = std::move(Flux2KleinPipeline::from_pretrained(runtime, "../utils/convert-model"));

    Flux2KleinPipeline::GenerationOptions options;
    options.prompt = "a lovely cat";

    auto images = pipeline(runtime, std::move(options));

    images[0].save("test.png");
}
