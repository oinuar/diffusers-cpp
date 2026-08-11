#include "ggml/Backend.hpp"
#include "ggml/Runtime.hpp"
#include "ggml/Scheduler.hpp"
#include "ggml/Computation.hpp"
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
    Scheduler scheduler({*cpu});
    Context context;
    Runtime runtime(scheduler, context);

    // initial value
    auto counter = runtime.create<int32_t>({1}, [](std::mt19937&) {
        return std::vector<int32_t>({42});
    });

    // counter + 1
    auto next_counter = counter + 1;
    next_counter = next_counter.copy(counter);
    
    Graph graph(runtime, {next_counter});
    std::vector<int32_t> result;

    // count to +10
    for (auto i = 0; i < 10; ++i) {
        Computation computation(graph);

        // read results on the last step
        if (i == 10 - 1)
            result = runtime.value<int32_t>(counter);
    }

    // print results
    for (auto& x : result)
        std::cout << "result: " << x << std::endl;

    /*std::filesystem::path path("../utils/convert-model/flux2-vae.gguf");

    AutoencoderKLFlux2 vae;

    GGUFLoaderVisitor loader(cpu, path);
    RethrowVisitor visitor(loader);
    vae.accept(visitor);
    visitor.rethrow();*/

    //Qwen3ForCausalLM::from_pretrained(cpu, Qwen3Config::from_file("../utils/convert-model/text-encoder/config.json"), "../utils/convert-model/qwen3.gguf");

    return 0;
}


#if 0
int main() {
    ggml_time_init();
    ggml_log_set([](ggml_log_level, const char* text, void*) { std::cerr << text; }, nullptr);

    ggml_backend_load_all();

    Backend cpu(GGML_BACKEND_DEVICE_TYPE_CPU);
    Scheduler scheduler({*cpu});
    Context context;
    Runtime runtime(scheduler, context);

    AutoencoderKLFlux2 vae(AutoencoderKLFlux2::Config::from_file("../utils/convert-model/vae/config.json"));
    std::filesystem::path path("../utils/convert-model/vae/flux2-vae.gguf");

    GGUFLoaderVisitor loader(cpu, path);
    RethrowVisitor visitor(loader);
    vae.accept(visitor);
    loader.validate();
    visitor.rethrow();

    //Qwen3ForCausalLM::from_pretrained(cpu, Qwen3Config::from_file("../utils/convert-model/text-encoder/config.json"), "../utils/convert-model/qwen3.gguf");

    return 0;
}
#endif