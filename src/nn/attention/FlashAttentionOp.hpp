#pragma once

#include "ggml/Tensor.hpp"
#include <optional>

class Scope;

struct FlashAttentionOp {
    Tensor operator ()(
        Scope scope,
        Tensor query,
        Tensor key,
        Tensor value,
        std::optional<Tensor> mask = std::nullopt,
        std::optional<float> scaling = std::nullopt);
};