#include "nn/attention/FlashAttentionOp.hpp"
#include "ggml/Runtime.hpp"

Tensor FlashAttentionOp::operator ()(
    Runtime& runtime,
    Tensor q,
    Tensor k,
    Tensor v,
    std::optional<Tensor> mask,
    std::optional<float> scaling)
{
    float scale = scaling.value_or(1.0f / std::sqrt((float)q.shape()[-1]));

    /*
     * ggml_flash_attn_ext() requires the mask to:
     *
     *   1. Have the full attention score shape:
     *
     *          (B, H, Q, K)
     *
     *      PyTorch allows broadcasting, for example:
     *
     *          (1, 1, Q, K)
     *          (B, 1, Q, K)
     *
     *      so expand the mask explicitly before passing it to GGML.
     *
     *   2. Be stored as F16.
     *
     *      GGML's flash attention kernel requires F16 masks even when
     *      q/k/v use another precision.
     */
    ggml_tensor* ggml_mask = nullptr;

    if (mask) {
        auto attention_shape = Tensor::Shape({
            q.shape()[0], // B: batch size
            q.shape()[1], // H: number of heads
            q.shape()[2], // Q: query sequence length
            k.shape()[2], // K: key sequence length
        });

        // Apply PyTorch-style broadcasting rules to make the mask explicit.
        auto target = Tensor::Shape::broadcast(mask->shape(), attention_shape);

        // Expand broadcast dimensions and convert to the format required
        // by ggml_flash_attn_ext().
        mask = mask->expand(target);
        mask = mask->to(GGML_TYPE_F16);

        ggml_mask = **mask;
    }

    /*
     * ggml_flash_attn_ext() returns output in GGML flash attention layout:
     *
     *     (B, S, H, D)
     *
     * while PyTorch scaled_dot_product_attention() returns:
     *
     *     (B, H, S, D)
     *
     * The Tensor wrapper needs the actual GGML output shape first, then we
     * permute the result back into the PyTorch-compatible layout.
     */
    auto ggml_flash_output_shape = q.shape(); // (B, H, S, D)
    std::swap(ggml_flash_output_shape[1], ggml_flash_output_shape[2]); // (B, S, H, D)

    /*
     * q/k/v use PyTorch-style logical shapes:
     *
     *     (B, H, S, D)
     *
     * The Tensor wrapper hides GGML's reversed ne[] storage order, so the
     * tensors can be passed directly to ggml_flash_attn_ext().
     *
     * Do not permute q/k/v here. The layout conversion is only needed for
     * the output of ggml_flash_attn_ext().
     */
    auto attn = Tensor(
        *runtime.context(),
        ggml_flash_attn_ext(
            *runtime.context(),
            *q,
            *k,
            *v,
            ggml_mask,
            scale,
            0.0f,
            0.0f),
        ggml_flash_output_shape
    );

    /* Convert GGML flash attention output:
     *
     *     (B, S, H, D)
     *
     * into PyTorch SDPA output:
     *
     *     (B, H, S, D)
     */
    return attn.permute({0, 2, 1, 3});
}
