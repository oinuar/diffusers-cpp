#include "ggml/Allocator.hpp"
#include "ggml/Context.hpp"
#include "ggml/Device.hpp"
#include "ggml/ExecutionRuntime.hpp"
#include <iostream>
#include <vector>

static std::string format_bytes(size_t bytes) {
    if (bytes >= 1024 * 1024) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "%.2f MB", double(bytes) / (1024.0 * 1024.0));
        return buf;
    }
    if (bytes >= 1024)
        return std::to_string(bytes / 1024) + " KB";
    return std::to_string(bytes) + " B";
}

static size_t count_tensors(ggml_context * ctx, ggml_backend_buffer_t buffer) {
    size_t n = 0;
    for (auto t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        if (t->buffer == buffer)
            ++n;
    }
    return n;
}

Allocator::Allocator(Device& device, const std::optional<ggml_backend_buffer_usage>& usage)
    : device_(device), usage_(usage)
{
}

void Allocator::allocate(Context& context) {
    auto buft = device_.buffer_type();
    
    // Calculate the total size required for unallocated tensors in the context.
    // This helper function calculates the size without actually allocating memory.
    size_t size = ggml_backend_alloc_ctx_tensors_from_buft_size(*context, buft);
    
    // If size is 0, there might be no tensors to allocate, or only views that need initialization
    if (size == 0) {
        for (auto t = ggml_get_first_tensor(*context); t != nullptr; t = ggml_get_next_tensor(*context, t)) {
            if (t->data == nullptr && t->view_src != nullptr && t->buffer == nullptr) {
                ggml_backend_view_init(t);
            }
        }
        return;
    }

    // 1. Manually allocate the buffer
    auto buff = ggml_backend_buft_alloc_buffer(buft, size);
    if (buff == nullptr)
        return;

    const char* usages[] = {
        " (any)",
        " (weights)",
        " (compute)"
    };

    std::cerr << "Allocating total of "
              << format_bytes(size)
              << " tensors to "
              << ggml_backend_dev_name(*device_)
              << (usage_ ? usages[*usage_] : "")
              << std::endl;

    // 2. Set the usage BEFORE initializing the tensors.
    // This is crucial for the Meta backend to see the correct usage (COMPUTE or WEIGHTS) 
    // when it initializes the tensors and derives split rules.
    if (usage_)
        ggml_backend_buffer_set_usage(buff, *usage_);

    // 3. Use ggml_tallocr to assign and initialize tensors into the pre-allocated buffer
    auto tallocr = ggml_tallocr_new(buff);
    size_t count = 0;

    for (auto t = ggml_get_first_tensor(*context); t != nullptr; t = ggml_get_next_tensor(*context, t)) {
        if (t->data == nullptr) {
            if (t->view_src == nullptr) {
                // Allocate standard tensor
                if (ggml_tallocr_alloc(&tallocr, t) != GGML_STATUS_SUCCESS) {
                    std::cerr << "Failed to allocate tensor " << ggml_get_name(t) << std::endl;
                    return;
                }
            } else if (t->buffer == nullptr) {
                // Initialize view tensor
                if (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                    std::cerr << "Failed to initialize view tensor " << ggml_get_name(t) << std::endl;
                    return;
                }
            }

            //std::cerr << "   - Allocated " << ggml_get_name(t) << ' ' << format_bytes(ggml_nbytes(t)) << std::endl;
            ++count;
        } else {
            if (t->view_src != nullptr && t->buffer == nullptr) {
                if (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                    std::cerr << "Failed to initialize pre-allocated view tensor " << ggml_get_name(t) << std::endl;
                    return;
                }
            }
        }
    }

    auto& buffer = buffers_.emplace_back(buff, usage_);

    std::cerr << "Allocated "
            << count
            << " tensors to a "
            << ggml_backend_dev_name(*device_)
            << " buffer of size "
            << format_bytes(ggml_backend_buffer_get_size(*buffer))
            << (usage_ ? usages[*usage_] : "")
            << std::endl;
}
