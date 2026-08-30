#pragma once

#include "ggml/Buffer.hpp"
#include <ggml.h>
#include <unordered_map>
#include <optional>

class Tensor;

class Allocator {
public:
    Allocator(ggml_backend_buffer_type_t buft);

    void reserve(const Tensor& tensor);
    void allocate(const ggml_backend_buffer_usage& usage = GGML_BACKEND_BUFFER_USAGE_ANY);
    bool contains(const Tensor& tensor) const;

private:
    struct Layout {
        size_t offset;
        size_t size;
    };

    ggml_backend_buffer_type_t buft_;
    std::unordered_map<ggml_tensor*, Layout> layout_;
    size_t total_size_;
    std::optional<Buffer> buffer_;
};
