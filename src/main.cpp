#include "ggml/Device.hpp"
#include "ggml/MetaDevice.hpp"
#include "ggml/Backend.hpp"
#include "ggml/Runtime.hpp"
#include "ggml/Scheduler.hpp"
#include "ggml/BufferAllocatorVisitor.hpp"
#include "diffusers/pipelines/flux2/Flux2KleinPipeline.hpp"
#include <iostream>
#include <filesystem>

int main() {
    ggml_time_init();
    ggml_log_set([](ggml_log_level, const char* text, void*) { std::cerr << text; }, nullptr);

    ggml_backend_load_all();

    auto gpus = MetaDevice::all(GGML_BACKEND_DEVICE_TYPE_GPU);
    Device cpu(GGML_BACKEND_DEVICE_TYPE_CPU);
    Backend meta_backend(gpus);
    Backend cpu_backend(cpu);
    Scheduler scheduler({&meta_backend, &cpu_backend}, 65536);
    Context context(65536);
    Runtime runtime(scheduler, context);
    BufferAllocatorVisitor weights_allocator(ggml_backend_dev_buffer_type(*gpus));

    auto pipeline = std::move(Flux2KleinPipeline::from_pretrained(runtime, "../utils/convert-model", &weights_allocator));

    Flux2KleinPipeline::GenerationOptions options;
    options.prompt = "a lovely cat";
    options.width = 256;
    options.height = 256;

    auto images = pipeline(runtime, std::move(options));

    images[0].save("test.png");
}
