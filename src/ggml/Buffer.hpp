#pragma once

#include <ggml.h>
#include <ggml-backend.h>
#include <stdexcept>

class Buffer {
public:
    Buffer(ggml_backend_buffer_type_t buft, size_t size)
        : buffer_(ggml_backend_buft_alloc_buffer(buft, size))
    {
        if (buffer_ == nullptr)
            throw std::runtime_error("Failed to allocate buffer of size " + std::to_string(size));
    }

    Buffer(Buffer&& other) : buffer_(other.buffer_) {
        other.buffer_ = nullptr;
    }

    ~Buffer() {
        if (buffer_ != nullptr)
            ggml_backend_buffer_free(buffer_);
    }

    ggml_backend_buffer_t operator *() {
        return buffer_;
    }

    Buffer(Buffer&) = delete;
    Buffer& operator =(const Buffer&) = delete;

private:
    ggml_backend_buffer_t buffer_;
};
