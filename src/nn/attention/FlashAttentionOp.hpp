#pragma once

#include "ggml/Tensor.hpp"
#include <optional>

class Runtime;

struct FlashAttentionOp {
    Tensor operator ()(
        Runtime& runtime,
        Tensor query,
        Tensor key,
        Tensor value,
        std::optional<Tensor> mask = std::nullopt,
        std::optional<float> scaling = std::nullopt);
};