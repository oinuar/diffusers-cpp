#pragma once

#include "ggml/Tensor.hpp"
#include <optional>

struct FlashAttentionOp {
    Tensor operator ()(
        ggml_context* ctx,
        Tensor query,
        Tensor key,
        Tensor value,
        std::optional<Tensor> mask = std::nullopt);
};