#include "nn/attention/SoftmaxAttentionOp.hpp"
#include "ggml/Context.hpp"
#include "ggml/Scope.hpp"
#include "ggml/Runtime.hpp"

Tensor SoftmaxAttentionOp::operator ()(
    Scope scope,
    Tensor query,
    Tensor key,
    Tensor value,
    std::optional<Tensor> mask,
    std::optional<float> scaling)
{
    float scale = scaling.value_or(1.0f / std::sqrt((float)query.shape()[-1]));

    auto q_shape = query.shape();

    /*
     * All tensors use the PyTorch layout (B, H, S, D). The key axis (S_k)
     * is the last logical dim of the scores, i.e. ggml axis 0, which is the
     * only axis the softmax may NOT be sharded along (each query needs the
     * full key range to normalize). sum_rows() enforces exactly that: it
     * carries the shard state over but refuses a shard of the reduced axis.
     */

    // scores = scale * k^T q. The project's mul_mat convention (lhs =
    // [in, out]): key plays the weight role with in = D, out = S_k; query
    // is the activation with out = S_q. The result's batch dims (H, B)
    // come from the activation.
    auto scores = Tensor(
        scope.runtime().mul_mat(*key, *query),
        Tensor::Shape({q_shape[0], q_shape[1], q_shape[2], key.shape()[2]}));

    scores = scores * scale;

    if (mask)
        scores = scores + *mask;   // (B, H, S_q, S_k), ggml_add broadcasts

    // Softmax over the key axis: exp / sum_k(exp). The max-subtraction of
    // the numerically stable form is omitted: softmax is shift-invariant,
    // and a max would reduce along the same key axis that no sharded state may
    // touch (max_rows is not a splittable op in the meta backend).
    auto e = exp(scores);
    auto attn_weights = e / e.sum(-1, /*keepdim=*/true);

    // out = v^T attn_weights. The reduction dim of mul_mat is the first raw
    // ggml axis (ne0), so the value operand must have its S_k axis there:
    // transpose (B, H, S_k, D) -> (B, H, D, S_k), i.e. raw ggml {S_k, D, H, B}.
    // ggml_mul_mat rejects a transposed lhs (nb0 > nb1), so the transpose
    // is materialized with a contiguous copy.
    auto v_t = value.transpose(2, 3).contiguous();
    return Tensor(
        scope.runtime().mul_mat(*v_t, *attn_weights),
        Tensor::Shape({q_shape[0], q_shape[1], q_shape[2], value.shape()[-1]}));
}
