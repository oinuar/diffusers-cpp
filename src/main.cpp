#if 0
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <chrono>
#include <thread>

// -----------------------------------------------------------------------------
// 50/50 tensor split
//
// For a tensor split along axis 0:
//
//   GPU 0 gets first half
//   GPU 1 gets second half
//
// ggml_backend_meta_split_state::ne is laid out as:
//   [segment0_dev0, segment0_dev1, ...]
// -----------------------------------------------------------------------------

static ggml_backend_meta_split_state split_50_50(
        const ggml_tensor * tensor,
        void * userdata) {
    (void) userdata;

    ggml_backend_meta_split_state state = {};

    state.axis = GGML_BACKEND_SPLIT_AXIS_0;
    state.n_segments = 1;
    state.nr[0] = 1;

    const int64_t n = tensor->ne[0];

    state.ne[0] = n / 2;
    state.ne[1] = n - state.ne[0];

    return state;
}

int main() {
    // -------------------------------------------------------------------------
    // Initialize ggml
    // -------------------------------------------------------------------------

    ggml_time_init();

    ggml_log_set(
        [](ggml_log_level, const char * text, void *) {
            std::fprintf(stderr, "%s", text);
        },
        nullptr);

    ggml_backend_load_all();

    // -------------------------------------------------------------------------
    // Find CUDA backend
    // -------------------------------------------------------------------------

    ggml_backend_reg_t cuda_reg = nullptr;

    for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);

        const char * name = ggml_backend_reg_name(reg);

        if (name && std::strcmp(name, "CUDA") == 0) {
            cuda_reg = reg;
            break;
        }
    }

    if (!cuda_reg) {
        std::fprintf(stderr, "CUDA backend not found\n");
        return 1;
    }

    // -------------------------------------------------------------------------
    // Get CUDA devices
    // -------------------------------------------------------------------------

    const size_t n_devs = ggml_backend_reg_dev_count(cuda_reg);

    std::printf("CUDA devices: %zu\n", n_devs);

    if (n_devs < 2) {
        std::fprintf(stderr, "Need at least 2 CUDA devices\n");
        return 1;
    }

    ggml_backend_dev_t devs[2];

    for (size_t i = 0; i < 2; ++i) {
        devs[i] = ggml_backend_reg_dev_get(cuda_reg, i);

        size_t free_mem = 0;
        size_t total_mem = 0;

        ggml_backend_dev_memory(
            devs[i],
            &free_mem,
            &total_mem);

        std::printf(
            "GPU %zu: %s\n"
            "       free:  %.2f GiB\n"
            "       total: %.2f GiB\n",
            i,
            ggml_backend_dev_name(devs[i]),
            free_mem / 1024.0 / 1024.0 / 1024.0,
            total_mem / 1024.0 / 1024.0 / 1024.0);
    }

    // -------------------------------------------------------------------------
    // Create META device
    //
    // This is the same mechanism used by llama.cpp tensor splitting.
    // -------------------------------------------------------------------------

    ggml_backend_dev_t meta_dev =
        ggml_backend_meta_device(
            devs,
            2,
            split_50_50,
            nullptr);

    if (!meta_dev) {
        std::fprintf(stderr, "Failed to create meta device\n");
        return 1;
    }

    std::printf(
        "Meta device: %s\n",
        ggml_backend_dev_name(meta_dev));

    // -------------------------------------------------------------------------
    // Get META buffer type
    //
    // This automatically contains the CUDA buffer type of both GPUs.
    // -------------------------------------------------------------------------

    ggml_backend_buffer_type_t buft =
        ggml_backend_dev_buffer_type(meta_dev);

    if (!buft) {
        std::fprintf(stderr, "Failed to get meta buffer type\n");
        return 1;
    }

    std::printf(
        "Buffer type: %s\n",
        ggml_backend_buft_name(buft));

    // -------------------------------------------------------------------------
    // Create a 20 GiB tensor.
    //
    // F32 = 4 bytes.
    //
    // With 50/50 split:
    //
    //     GPU0 ~10 GiB
    //     GPU1 ~10 GiB
    //
    // This fits your 16 GiB + 12 GiB cards.
    // -------------------------------------------------------------------------

    constexpr size_t GiB = 1024ull * 1024ull * 1024ull;

    constexpr size_t tensor_size = 20ull * GiB;

    const int64_t n_elements =
        tensor_size / sizeof(float);

    ggml_init_params params = {
        .mem_size   = 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };

    ggml_context * ctx = ggml_init(params);

    if (!ctx) {
        std::fprintf(stderr, "ggml_init() failed\n");
        return 1;
    }

    ggml_tensor * tensor =
        ggml_new_tensor_1d(
            ctx,
            GGML_TYPE_F32,
            n_elements);

    if (!tensor) {
        std::fprintf(stderr, "Failed to create tensor\n");
        ggml_free(ctx);
        return 1;
    }

    std::printf(
        "Tensor: %.2f GiB, ne[0] = %lld\n",
        ggml_nbytes(tensor) / (double) GiB,
        (long long) tensor->ne[0]);

    // -------------------------------------------------------------------------
    // Allocate through META buffer.
    // -------------------------------------------------------------------------

    std::printf("\nAllocating...\n");

    ggml_backend_buffer_t buffer =
        ggml_backend_alloc_ctx_tensors_from_buft(
            ctx,
            buft);

    if (!buffer) {
        std::fprintf(stderr, "\nFAILED\n");
        ggml_free(ctx);
        return 1;
    }

    std::printf("\nSUCCESS!\n");

    std::printf(
        "Meta buffer size: %.2f GiB\n",
        ggml_backend_buffer_get_size(buffer) /
            (double) GiB);

    std::this_thread::sleep_for(std::chrono::seconds(10));

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);

    return 0;
}

#endif

#include "ggml/Device.hpp"
#include "ggml/MetaDevice.hpp"
#include "ggml/Backend.hpp"
#include "ggml/Context.hpp"
#include "ggml/DeviceAllocator.hpp"
#include "ggml/Scheduler.hpp"
#include "diffusers/pipelines/flux2/Flux2KleinPipeline.hpp"
#include <iostream>
#include <filesystem>
#include <chrono>
#include <thread>


int main() {
    ggml_time_init();
    ggml_log_set([](ggml_log_level, const char* text, void*) { std::cerr << text; }, nullptr);

    ggml_backend_load_all();

    auto gpus = MetaDevice::all(GGML_BACKEND_DEVICE_TYPE_GPU);
    Device cpu(GGML_BACKEND_DEVICE_TYPE_CPU);
    Backend gpus_backend(gpus);
    Backend cpu_backend(cpu);
    Scheduler scheduler({&gpus_backend, &cpu_backend}, 836464);
    Context weights_context(836464);
    DeviceAllocator allocator(weights_context, gpus);

    auto pipeline = std::move(Flux2KleinPipeline::from_pretrained(weights_context, weights_context, weights_context, "../utils/convert-model"));

    allocator.allocate(GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    
    constexpr size_t GiB = 1024ull * 1024ull * 1024ull;
    
    std::printf(
        "Meta buffer size: %.2f GiB\n",
        allocator.buffer().size() /
            (double) GiB);

    Flux2KleinPipeline::GenerationOptions options;
    options.prompt = "a lovely cat";
    options.width = 256;
    options.height = 256;

    auto images = pipeline(scheduler, weights_context, weights_context, weights_context, gpus, std::move(options));

    images[0].save("test2.png");
}

/*

int main() {
    ggml_time_init();
    ggml_log_set([](ggml_log_level, const char* text, void*) { std::cerr << text; }, nullptr);

    ggml_backend_load_all();

    auto gpus = MetaDevice::all(GGML_BACKEND_DEVICE_TYPE_GPU);
    Device cpu(GGML_BACKEND_DEVICE_TYPE_CPU);
    Backend meta_backend(gpus);
    Backend cpu_backend(cpu);
    Scheduler scheduler({&meta_backend, &cpu_backend});
    Context context(scheduler.capacity());
    Runtime runtime(scheduler, context);

    // initial value
    auto counter = context.create<float>({1}, [](std::mt19937&) {
        return std::vector<float>({42});
    });

    // counter + 1
    auto next_counter = counter + 1.0f;
    
    Graph graph(runtime, {next_counter});
    std::vector<float> result;

    // count to +10
    for (auto i = 0; i < 10; ++i) {
        Computation computation(graph);

        // assign counter <- counter+1
        runtime.copy(computation.results().at(0), counter);

        // read results on the last step
        if (i == 10 - 1)
            result = context.value<float>(counter);
    }

    // print results
    for (auto& x : result)
        std::cout << "result: " << x << std::endl;
}

*/