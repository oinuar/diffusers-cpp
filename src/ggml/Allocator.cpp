#include "ggml/Allocator.hpp"
#include "ggml/Context.hpp"
#include "ggml/Device.hpp"
#include "ggml/ExecutionRuntime.hpp"

Allocator::Allocator(Device& device, const std::optional<ggml_backend_buffer_usage>& usage)
    : device_(device), usage_(usage)
{
}

void Allocator::allocate(Context& context) {
    auto buft = device_.buffer_type();
    auto buffer = ggml_backend_alloc_ctx_tensors_from_buft(*context, buft);

    // NULL is returned when every tensor in the context already has a buffer,
    // e.g. when the graph is built entirely in another context. There is
    // nothing to allocate in that case.
    if (buffer == nullptr)
        return;

    buffers_.emplace_back(buffer, usage_);
}
