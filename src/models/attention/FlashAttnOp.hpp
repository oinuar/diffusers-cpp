#pragma once

#include "Tensor.hpp"

static bool ggml_ext_is_padded_1d(const ggml_tensor* x) {
    return x->nb[0] == ggml_type_size(x->type) &&
           x->nb[2] == x->nb[1] * x->ne[1] &&
           x->nb[3] == x->nb[2] * x->ne[2];
}

static ggml_tensor* ggml_ext_scale(ggml_context* ctx,
                                              ggml_tensor* x,
                                              float factor,
                                              bool inplace = false) {
    if (!ggml_ext_is_padded_1d(x)) {
        x = ggml_cont(ctx, x);
    }
    if (inplace) {
        x = ggml_scale_inplace(ctx, x, factor);
    } else {
        x = ggml_scale(ctx, x, factor);
    }
    return x;
}

// FlashAttnOp — lines 1339-1392 of ggml_extend.hpp      
struct FlashAttnOp {
     Tensor operator()(ggml_context* ctx, ggml_tensor* q, ggml_tensor* k,
                         ggml_tensor* v, int64_t n_head, int64_t d_head,
                         int64_t n_kv_head, int64_t L_q, int64_t L_k,
                         int64_t N /* batch */, ggml_tensor* mask = nullptr, float kv_scale    = 1.0f) const
     {
          float scale = (1.0f / sqrt((float)d_head));

          if (kv_scale != 1.0f)
               k = ggml_ext_scale(ctx, k, kv_scale);

          k = ggml_cast(ctx, k, GGML_TYPE_F16);

          v = ggml_ext_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
          v = ggml_reshape_3d(ctx, v, d_head, L_k, n_kv_head * N);

          if (kv_scale != 1.0f)
               v = ggml_ext_scale(ctx, v, kv_scale);

          v = ggml_cast(ctx, v, GGML_TYPE_F16);
          
          if (mask != nullptr) {
               // ggml_flash_attn_ext expects the mask as a contiguous F16 tensor shaped
               // [n_kv, n_q, (heads), (batch)] (ne0 = key length, ne1 = query length) and,
               // unlike the manual-attention path, does not broadcast the query dimension.
               // Some callers (e.g. Chroma/T5) pass a per-key padding mask broadcast over
               // queries ([n_kv, 1, ...]); materialize the query dimension to L_q so the
               // kernel indexes it correctly. (A bare ggml_transpose here produced a
               // [1, n_kv, ...] mask that the kernel silently misreads, yielding NaN/blank
               // output for masked flash attention.)
               if (mask->ne[1] != L_q) {
                    mask = ggml_repeat(ctx, mask,
                              ggml_new_tensor_4d(ctx, mask->type, mask->ne[0], L_q,
                                             mask->ne[2], mask->ne[3]));
               }
               mask = ggml_cast(ctx, mask, GGML_TYPE_F16);
          }

          auto kqv = ggml_flash_attn_ext(ctx, q, k, v, mask, scale / kv_scale, 0, 0);
          ggml_flash_attn_ext_set_prec(kqv, GGML_PREC_F32);

          if (kv_scale != 1.0f)
               kqv = ggml_ext_scale(ctx, kqv, 1.0f / kv_scale);

          if (kqv == nullptr)
               throw std::runtime_error("FlashAttnOp: failed, maybe backend does not support flash attention? Try SoftmaxAttnOp instead.");

          kqv = ggml_view_3d(ctx, kqv, d_head, n_head, L_q, kqv->nb[1], kqv->nb[2], 0);
          kqv = ggml_ext_cont(ctx, kqv);
          kqv = ggml_reshape_3d(ctx, kqv, d_head * n_head, L_q, N);  // [N, L_q, C]

          return Tensor(ctx, kqv);
     }
};
