#pragma once

#include "ggml/Buffer.hpp"
#include <ggml.h>
#include <vector>
#include <optional>

class Tensor;

class Allocator {
public:
    Allocator(ggml_backend_buffer_type_t buft);

    void reserve(const Tensor& tensor);
    void allocate(const ggml_backend_buffer_usage& usage = GGML_BACKEND_BUFFER_USAGE_ANY);

private:
    struct Layout {
        ggml_tensor* tensor;
        size_t offset;
        size_t size;
    };

    ggml_backend_buffer_type_t buft_;
    std::vector<Layout> layout_;
    size_t total_size_;
    std::optional<Buffer> buffer_;
};
