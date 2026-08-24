#include "ggml/BufferAllocatorVisitor.hpp"
#include "nn/Parameter.hpp"

BufferAllocatorVisitor::BufferAllocatorVisitor(ggml_backend_buffer_type_t buft)
    : buft_(buft), layout_(), total_size_(0), buffer_()
{
}

void BufferAllocatorVisitor::visit(Parameter& parameter, std::vector<std::string> path) {
    // Size required by this backend for this tensor
    auto size = ggml_backend_buft_get_alloc_size(buft_, **parameter);

    // Align the start of the next tensor
    auto alignment = ggml_backend_buft_get_alignment(buft_);

    total_size_ = (total_size_ + alignment - 1) / alignment * alignment;

    layout_.push_back({**parameter, total_size_, size});

    total_size_ += size;
}

void BufferAllocatorVisitor::allocate() {
    // Already allocated
    if (buffer_)
        return;

    if (total_size_ == 0 || layout_.empty())
        throw std::runtime_error("Nothing to allocate. Did you forget to visit() module tree first?");

    buffer_.emplace(buft_, total_size_);

    auto base = ggml_backend_buffer_get_base(**buffer_);

    // Allocate tensors in buffer
    for (auto& entry : layout_) {
        auto addr = reinterpret_cast<char*>(base) + entry.offset;

        if (ggml_backend_tensor_alloc(**buffer_, entry.tensor, addr) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("Failed to allocate tensor in buffer");
    }

    layout_.clear();
    total_size_ = 0;
}
