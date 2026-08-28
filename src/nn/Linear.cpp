#include "nn/Linear.hpp"
#include "nn/Parameter.hpp"
#include "ggml/Runtime.hpp"
#include <vector>

namespace {

/*
 * Maximum number of rows (the second-to-last input dimension) per f32
 * ggml_mul_mat on the GPU backend.
 *
 * ggml's CUDA backend unconditionally configures its cuBLAS handles with
 * CUBLAS_TF32_TENSOR_OP_MATH (ggml/src/ggml-cuda/common.cuh, cublas_handle()).
 * On NVIDIA GPUs with compute capability >= 8.0 every f32 GEMM that is
 * dispatched to cuBLAS is therefore downcast to TF32 (11-bit mantissa), which
 * drifts by ~1e-3 relative to the f32 CPU path and breaks this project's
 * f32 parity with PyTorch.
 *
 * The custom f32 vector kernel (mmvf) is full precision. Its dispatch
 * (ggml_cuda_should_use_mmvf, ggml/src/ggml-cuda/mmvf.cu) accepts at most:
 *   - 3 rows on NVIDIA Ampere+ and pre-Turing NVIDIA,
 *   - 4 rows on Turing,
 *   - 3 or 8 rows on AMD,
 * so three rows is the maximum that stays on the exact f32 path on every
 * supported device. Larger batches are split into chunks of that size and
 * concatenated, which keeps the result identical to a single f32 GEMM.
 */
constexpr int64_t kMaxExactF32MatmulRows = 3;

} // namespace
Linear::Linear(
    int64_t in_features,
    int64_t out_features,
    bool bias
) : bias_(bias)
{
    // The weight (and bias) shard their output features across the devices of a
    // meta device, so the ggml_mul_mat in forward computes a slice of the output
    // rows per device with zero communication.
    auto weight = std::make_shared<Parameter>(Tensor::Shape({out_features, in_features}));
    modules["weight"] = weight;

    if (bias_) {
        auto bias = std::make_shared<Parameter>(Tensor::Shape({out_features}));
        modules["bias"] = bias;
    }
}

Tensor Linear::forward(Runtime& runtime, Tensor x) {
    // Setup matching split for input Tensor
    //runtime.split(x, 1);

    auto weight = std::static_pointer_cast<Parameter>(modules["weight"])->forward();

    // Number of GEMM rows is the size of the second-to-last input dimension:
    //
    //     x: [..., S, F]  (PyTorch)  ->  ne = {F, S, ...}  (ggml)
    //
    // and ggml_cuda_should_use_mmvf() gates the exact f32 vector kernel on
    // ne[1] (S) only.
    const int64_t row_dim = x.shape().rank() - 2;
    const int64_t rows = x.shape()[row_dim];

    /*
     * Weight is logically shaped [out_features, in_features] (PyTorch),
     * but stored in GGML's native reversed layout {in_features, out_features}.
     * Since ggml_mul_mat() already performs Aᵀ * B on its first operand,
     * explicitly transposing the weight would transpose it twice.
     */
    auto mul_mat = [&](const Tensor& in) -> Tensor {
        Tensor::Shape out_shape = in.shape();
        out_shape[out_shape.rank() - 1] = weight.shape()[0];
        return Tensor(*runtime.context(), ggml_mul_mat(*runtime.context(), *weight, *in), out_shape);
    };

    Tensor y;

    if (rows <= kMaxExactF32MatmulRows) {
        y = mul_mat(x);
    } else {
        // Chunk the rows so every mul_mat stays on the exact f32 kernel path
        // (see kMaxExactF32MatmulRows) instead of the TF32 cuBLAS fallback.
        std::vector<Tensor> parts;
        parts.reserve((rows + kMaxExactF32MatmulRows - 1) / kMaxExactF32MatmulRows);

        for (const auto& chunk : x.split(kMaxExactF32MatmulRows, row_dim))
            parts.push_back(mul_mat(chunk));

        y = Tensor::cat(parts, row_dim);
    }

    if (bias_) {
        auto bias = std::static_pointer_cast<Parameter>(modules["bias"])->forward();

        y = Tensor(*runtime.context(), ggml_add(*runtime.context(), *y, *bias), y.shape());
    }

    return y;
}
