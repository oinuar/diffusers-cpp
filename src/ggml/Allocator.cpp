#include "ggml/Allocator.hpp"
#include "ggml/Context.hpp"
#include "ggml/Device.hpp"
#include "ggml/ExecutionRuntime.hpp"
#include <iostream>

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
    for (auto t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t))
        if (t->buffer == buffer)
            ++n;
    return n;
}

Allocator::Allocator(Device& device, const std::optional<ggml_backend_buffer_usage>& usage)
    : device_(device), usage_(usage)
{
}

void Allocator::allocate(Context& context) {
    auto buft = device_.buffer_type();
    auto buff = ggml_backend_alloc_ctx_tensors_from_buft(*context, buft);

    // NULL is returned when every tensor in the context already has a buffer,
    // e.g. when the graph is built entirely in another context. There is
    // nothing to allocate in that case.
    if (buff == nullptr)
        return;

    auto& buffer = buffers_.emplace_back(buff, usage_);

    std::cerr << "allocated "
            << count_tensors(*context, *buffer)
            << " tensors to a "
            << ggml_backend_dev_name(*device_)
            << " buffer of size "
            << format_bytes(ggml_backend_buffer_get_size(*buffer))
            << std::endl;

}
