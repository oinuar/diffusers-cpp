#pragma once

#include "ggml.h"

#include <vector>

class Arena {
public:
    Arena(size_t graph_size) : buffer_() {
        size_t buf_size = ggml_tensor_overhead()*graph_size + ggml_graph_overhead();
        buffer_.resize(buf_size);
    }

    std::byte* data() {
        return buffer_.data();
    }

    size_t size() const {
        return buffer_.size();
    }

private:
    std::vector<std::byte> buffer_;
};
