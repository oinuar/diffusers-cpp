#include "ggml/Allocator.hpp"
#include "ggml/Context.hpp"
#include "ggml/Device.hpp"
#include "ggml/ExecutionRuntime.hpp"
#include <iostream>
#include <vector>

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

    // Use the backend-aware allocation function. For the Meta backend, this 
    // automatically applies the split rules and allocates memory on each 
    // underlying device proportionally.
    auto buff = ggml_backend_alloc_ctx_tensors_from_buft(*context, buft);
    if (buff == nullptr)
        return;

    // Set the usage on the Meta buffer. The Meta backend will automatically 
    // propagate this usage to all its underlying simple device buffers.
    if (usage_)
        ggml_backend_buffer_set_usage(buff, *usage_);

    // Initialize any views whose source is in a different context.
    // Views within the same context are already initialized by alloc_buffer_n.
    for (auto t = ggml_get_first_tensor(*context); t != nullptr; t = ggml_get_next_tensor(*context, t)) {
        if (t->data == nullptr && t->view_src != nullptr && t->buffer == nullptr) {
            if (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                std::cerr << "Failed to initialize view tensor " << ggml_get_name(t) << std::endl;
                return;
            }
        }
    }

    size_t count = count_tensors(*context, buff);
    auto it = buffers_.emplace(*context, std::move(Buffer(buff, usage_)));

    const char* usages[] = {
        " (any)",
        " (weights)",
        " (compute)"
    };

    std::cerr << "Allocated "
              << count
              << " tensors to a "
              << ggml_backend_dev_name(*device_)
              << " buffer of size "
              << format_bytes(ggml_backend_buffer_get_size(*it->second))
              << (usage_ ? usages[*usage_] : "")
              << std::endl;
}

void Allocator::deallocate(Context& context) {
    auto range = buffers_.equal_range(*context);

    // Nothing was allocated for this context (e.g. an empty scope).
    if (range.first == range.second)
        return;

    auto bytes = 0.0;
    std::vector<ggml_backend_buffer_t> freed;
    for (auto it = range.first; it != range.second; ++it) {
        bytes += ggml_backend_buffer_get_size(*it->second);
        freed.push_back(*it->second);
    }

    // Un-assign the tensors of the freed buffers: they keep their
    // metadata in the context, but their storage is gone. Resetting the
    // pointers both prevents any use of the freed storage and lets the
    // context be allocated again later (the ggml allocators skip
    // tensors that still carry data).
    for (auto t = ggml_get_first_tensor(*context); t != nullptr; t = ggml_get_next_tensor(*context, t)) {
        for (auto buffer : freed)
            if (t->buffer == buffer) {
                t->data = nullptr;
                t->buffer = nullptr;
                break;
            }
    }

    auto count = buffers_.erase(*context);

    std::cerr << "Deallocated "
              << count
              << " buffers from "
              << ggml_backend_dev_name(*device_)
              << " of size "
              << format_bytes(bytes)
              << std::endl;
}

std::string Allocator::format_bytes(size_t bytes) {
    if (bytes >= 1024 * 1024 * 1024) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "%.2f GB", double(bytes) / (1024.0 * 1024.0 * 1024.0));
        return buf;
    }
    if (bytes >= 1024 * 1024) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "%.2f MB", double(bytes) / (1024.0 * 1024.0));
        return buf;
    }
    if (bytes >= 1024)
        return std::to_string(bytes / 1024) + " KB";
    return std::to_string(bytes) + " B";
}
