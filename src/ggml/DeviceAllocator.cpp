#include "ggml/DeviceAllocator.hpp"
#include "ggml/Tensor.hpp"
#include "ggml/Device.hpp"
#include "ggml/Context.hpp"

DeviceAllocator::DeviceAllocator(Context& context, const Device& device)
    : context_(context), buft_(ggml_backend_dev_buffer_type(*device)), buffers_()
{
}

void DeviceAllocator::allocate(const std::optional<ggml_backend_buffer_usage>& usage) {
    auto buffer = ggml_backend_alloc_ctx_tensors_from_buft(*context_, buft_);

    // NULL is returned when every tensor in the context already has a buffer,
    // e.g. when the graph is built entirely in another context. There is
    // nothing to allocate in that case.
    if (buffer == nullptr)
        return;

    buffers_.emplace_back(buffer, usage);
}

void DeviceAllocator::reset() {
    buffers_.clear();
    context_.reset();
}
