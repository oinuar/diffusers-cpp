#pragma once

#include "ggml/Tensor.hpp"

struct FlashAttnOp {
     Tensor operator()(ggml_context* ctx, ggml_tensor* q, ggml_tensor* k,
                         ggml_tensor* v, int64_t n_head, int64_t d_head,
                         int64_t n_kv_head, int64_t L_q, int64_t L_k,
                         int64_t N /* batch */, ggml_tensor* mask = nullptr, float kv_scale    = 1.0f) const;
};
