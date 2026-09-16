#pragma once

#include "ggml/Tensor.hpp"
#include <optional>

class Scope;

// Scaled dot-product attention built from splittable ggml primitives
// (mul_mat / scale / exp / sum_rows / div) instead of ggml_flash_attn_ext.
//
// The tensor-parallel meta backend can only run flash_attn_ext with q/k/v
// sharded along the sequence axis and the output sharded along the heads
// axis, so a single attention head cannot be planned on more than one
// device. Every primitive this operator uses carries the shard state of its
// input, so the planner may keep the attention fully replicated or shard it
// along the sequence axis regardless of the head count.
struct SoftmaxAttentionOp {
    Tensor operator ()(
        Scope scope,
        Tensor query,
        Tensor key,
        Tensor value,
        std::optional<Tensor> mask = std::nullopt,
        std::optional<float> scaling = std::nullopt);
};
