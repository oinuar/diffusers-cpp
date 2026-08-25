#pragma once

#include "nn/Visitor.hpp"
#include "ggml/Buffer.hpp"
#include <ggml.h>
#include <vector>
#include <optional>

class BufferAllocatorVisitor : public Visitor {
public:
    BufferAllocatorVisitor(ggml_backend_buffer_type_t buft);

    virtual void visit(Parameter& parameter, std::vector<std::string> path);

    void allocate();

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
