#include "ggml/Backend.hpp"
#include "ggml/Runtime.hpp"
#include "ggml/Scheduler.hpp"
#include "ggml/Allocator.hpp"
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

    Device cpu(GGML_BACKEND_DEVICE_TYPE_CPU);
    Backend cpu_backend(cpu);
    Scheduler scheduler({&cpu_backend}, 65536);
    Context weights_context(65536);
    Context context(65536);

    auto pipeline = std::move(Flux2KleinPipeline::from_pretrained(weights_context, weights_context, weights_context, "../utils/convert-model"));

    Flux2KleinPipeline::GenerationOptions options;
    options.prompt = "a lovely cat";
    options.width = 256;
    options.height = 256;

    /*auto images = pipeline(allocator, scheduler, context, weights_context, weights_context, weights_context, std::move(options));

    images[0].save("test.png");*/
}
