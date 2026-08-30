#include "ggml/Allocator.hpp"
#include "ggml/Tensor.hpp"

Allocator::Allocator(ggml_backend_buffer_type_t buft)
    : buft_(buft), layout_(), total_size_(0), buffer_(std::nullopt)
{
}

void Allocator::reserve(const Tensor& tensor) {
    if (buffer_)
        throw std::runtime_error("Buffer is already allocated");

    if (contains(tensor))
        throw std::runtime_error("Tensor already exists");

    // Size required by this backend for this tensor
    auto size = ggml_backend_buft_get_alloc_size(buft_, *tensor);

    // Align the start of the next tensor
    auto alignment = ggml_backend_buft_get_alignment(buft_);

    total_size_ = (total_size_ + alignment - 1) / alignment * alignment;

    layout_[*tensor] = {total_size_, size};

    total_size_ += size;
}

void Allocator::allocate(const ggml_backend_buffer_usage& usage) {
    // Nothing to do if already allocated
    if (buffer_)
        return;

    buffer_.emplace(buft_, total_size_, usage);

    auto base = ggml_backend_buffer_get_base(**buffer_);

    // Allocate tensors in buffer
    for (auto& [tensor, entry] : layout_) {
        auto addr = reinterpret_cast<char*>(base) + entry.offset;

        if (ggml_backend_tensor_alloc(**buffer_, tensor, addr) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("Failed to allocate tensor in buffer");
    }

    layout_.clear();
    total_size_ = 0;
}

bool Allocator::contains(const Tensor& tensor) const {
    return layout_.find(*tensor) != std::end(layout_);
}
