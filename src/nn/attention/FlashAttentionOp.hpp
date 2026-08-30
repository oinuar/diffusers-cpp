#pragma once

#include "ggml/Tensor.hpp"
#include <optional>

class Context;

struct FlashAttentionOp {
    Tensor operator ()(
        Context& context,
        Tensor query,
        Tensor key,
        Tensor value,
        std::optional<Tensor> mask = std::nullopt,
        std::optional<float> scaling = std::nullopt);
};