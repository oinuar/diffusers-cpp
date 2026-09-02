#include "ggml/DeviceAllocator.hpp"
#include "ggml/Tensor.hpp"
#include "ggml/Device.hpp"
#include "ggml/Context.hpp"

DeviceAllocator::DeviceAllocator(Context& context, const Device& device)
    : context_(context), buft_(ggml_backend_dev_buffer_type(*device)), buffer_(std::nullopt)
{
}

void DeviceAllocator::allocate(const std::optional<ggml_backend_buffer_usage>& usage) {
    if (buffer_)
        throw new std::runtime_error("Already allocated");

    auto buffer = ggml_backend_alloc_ctx_tensors_from_buft(*context_, buft_);

    buffer_.emplace(buffer, usage);
}
