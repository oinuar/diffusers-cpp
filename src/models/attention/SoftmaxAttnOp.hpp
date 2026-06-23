#pragma once

#include "Tensor.hpp"

static ggml_tensor* ggml_ext_cont(ggml_context* ctx,
                                             ggml_tensor* x) {
    if (ggml_is_contiguous(x)) {
        return x;
    }
    return ggml_cont(ctx, x);
}


// Mirrors the softmax fallback path in ggml_ext_attention_ext()
// Lines 1394–1407 of ggml_extend.hpp
struct SoftmaxAttnOp {
    Tensor operator()(ggml_context* ctx, ggml_tensor* q, ggml_tensor* k,
                         ggml_tensor* v, int64_t n_head, int64_t d_head,
                         int64_t n_kv_head, int64_t L_q, int64_t L_k,
                         int64_t N /* batch */, ggml_tensor* mask = nullptr) const noexcept
     {
          float scale = (1.0f / sqrt((float)d_head));

          v = ggml_ext_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));  // [N, n_kv_head, d_head, L_k]
          v = ggml_reshape_3d(ctx, v, L_k, d_head, n_kv_head * N);   // [N * n_kv_head, d_head, L_k]

          auto kq = ggml_mul_mat(ctx, k, q);  // [N * n_head, L_q, L_k]
          ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
          kq = ggml_scale_inplace(ctx, kq, scale);
          if (mask)
               kq = ggml_add_inplace(ctx, kq, mask);
          kq = ggml_soft_max_inplace(ctx, kq);

          auto kqv = ggml_mul_mat(ctx, v, kq);  // [N * n_head, L_q, d_head]

          kqv = ggml_reshape_4d(ctx, kqv, d_head, L_q, n_head, N);  // [N, n_head, L_q, d_head]
          kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);                 // [N, L_q, n_head, d_head]

          kqv = ggml_ext_cont(ctx, kqv);
          kqv = ggml_reshape_3d(ctx, kqv, d_head * n_head, L_q, N);  // [N, L_q, C]

          return Tensor(ctx, kqv);
     }
};
