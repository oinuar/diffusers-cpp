#pragma once

#include <ggml.h>
#include <ggml-backend.h>
#include <stdexcept>
#include <optional>

class Buffer {
public:
    Buffer(ggml_backend_buffer_t buffer, const std::optional<ggml_backend_buffer_usage>& usage = std::nullopt)
        : buffer_(buffer)
    {
        if (buffer_ == nullptr)
            throw std::invalid_argument("'buffer' is null");

        if (usage)
            ggml_backend_buffer_set_usage(buffer_, *usage);
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

    size_t size() const {
        return ggml_backend_buffer_get_size(buffer_);
    }

    Buffer(Buffer&) = delete;
    Buffer& operator =(const Buffer&) = delete;

private:
    ggml_backend_buffer_t buffer_;
};
