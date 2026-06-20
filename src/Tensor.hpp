#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <vector>
#include <cmath>

#include "ggml.h"

/**
 * Lightweight C++ wrapper around `ggml_tensor*` with a torch.Tensor-like API.
 *
 */
class Tensor {
public:
    Tensor() = default;

    Tensor(ggml_context* ctx, ggml_tensor* t)
        : ctx_(ctx), t_(t) {}

    int ndim() const {
        return ggml_n_dims(t_);
    }

    int64_t numel() const {
        return ggml_nelements(t_);
    }

    ggml_type dtype() const {
        return t_->type;
    }

    std::array<int64_t, 4> shape() const {
        return { t_->ne[0], t_->ne[1], t_->ne[2], t_->ne[3] };
    }

    ggml_tensor* operator *() const { return t_; }


    template <size_t N>
    static Tensor empty(
        ggml_context* ctx,
        ggml_type type,
        std::array<int64_t, N> shape)
    {
        return Tensor(ctx, ggml_new_tensor(ctx, type, N, shape.data()));
    }

    static Tensor scalar(
        ggml_context* ctx,
        float value,
        ggml_type type = GGML_TYPE_F32)
    {
        auto tensor = empty<1>(ctx, GGML_TYPE_F32, {1});
        tensor.t_ = ggml_fill_inplace(tensor.ctx_, tensor.t_, value);

        return tensor;
    }

    template <size_t N>
    static Tensor zeros(
        ggml_context* ctx,
        std::array<int64_t, N> shape,
        ggml_type type = GGML_TYPE_F32)
    {
        auto tensor = empty<N>(ctx, type, shape);
        tensor.t_ = ggml_fill_inplace(ctx, tensor.t_, 0.0f);
        return tensor;
    }

    template <size_t N>
    static Tensor ones(
        ggml_context* ctx,
        std::array<int64_t, N> shape,
        ggml_type type = GGML_TYPE_F32)
    {
        auto tensor = empty<N>(ctx, type, shape);
        tensor.t_ = ggml_fill_inplace(ctx, tensor.t_, 1.0f);
        return tensor;
    }

    static Tensor arange(
        ggml_context* ctx,
        float start,
        float stop,
        float step = 1.0f)
    {
        return Tensor(ctx, ggml_arange(ctx, start, stop, step));
    }

    /// Concates tensors along dimension `dim`.
    static Tensor cat(const std::vector<Tensor>& tensors, int dim);


    template <size_t N>
    Tensor reshape(std::array<int64_t, N> shape) const;

    /// Permute all four dimensions according to `order` [d0, d1, d2, d3].
    /// Negative indices are supported. Dimensions beyond ndim() are clamped.
    template <size_t N>
    Tensor permute(std::array<int, N> order) const;

    Tensor squeeze() const;

    Tensor unsqueeze(int dim) const;

    /// Narrow a dimension: keep elements [start, start+length) along `dim`.
    Tensor narrow(int dim, int64_t start, int64_t length) const;

    /// Chunk a tensor into `n` roughly equal pieces along dimension `dim`.
    std::vector<Tensor> chunk(int n, int dim = 0) const;

    /// Split a tensor into chunks of size `split_size` along dimension `dim`.
    std::vector<Tensor> split(int64_t split_size, int dim = 0) const;

    Tensor operator -() const {
        return Tensor(ctx_, ggml_neg(ctx_, t_));
    }

    Tensor operator+(const Tensor & rhs) const {
        return Tensor(ctx_, ggml_add(ctx_, t_, rhs.t_));
    }

    Tensor operator-(const Tensor & rhs) const {
        return Tensor(ctx_, ggml_sub(ctx_, t_, rhs.t_));
    }

    Tensor operator*(const Tensor & rhs) const {
        return Tensor(ctx_, ggml_mul(ctx_, t_, rhs.t_));
    }

    Tensor operator/(const Tensor & rhs) const {
        return Tensor(ctx_, ggml_div(ctx_, t_, rhs.t_));
    }

    Tensor operator*(float rhs) const {
        return Tensor(ctx_, ggml_scale(ctx_, t_, rhs));
    }

    Tensor operator+(float rhs) const {
        return *this + scalar(ctx_, rhs);
    }

    Tensor pow(float exponent) {
        // exp(exponent * log(x))  — works for arbitrary real exponents
        return exp(log(*this) * exponent);
    }

    Tensor clip(float a, float b) const {
        return Tensor(ctx_, ggml_clamp(ctx_, t_, a, b));
    }

    Tensor sum() const {
        return Tensor(ctx_, ggml_sum(ctx_, t_));
    }

    Tensor mean(/*TODO: dim?*/) const {
        return sum() * (1.0f / (float)numel());
    }
    
private:
    ggml_context* ctx_;
    ggml_tensor* t_;

    friend Tensor operator-(float value, const Tensor& tensor);
    friend Tensor operator/(float value, const Tensor& tensor);
    friend Tensor abs(const Tensor& tensor);
    friend Tensor sqrt(const Tensor& tensor);
    friend Tensor exp(const Tensor& tensor);
    friend Tensor log(const Tensor& tensor);
    friend Tensor sin(const Tensor& tensor);
    friend Tensor cos(const Tensor& tensor);
};

inline Tensor operator+(float value, const Tensor& tensor) {
    return tensor + value;
}

inline Tensor operator-(float value, const Tensor& tensor) {
    return Tensor::scalar(tensor.ctx_, value) - tensor;
}

inline Tensor operator*(float value, const Tensor& tensor) {
    return tensor * value;
}

inline Tensor operator/(float value, const Tensor& tensor) {
    return Tensor::scalar(tensor.ctx_, value) / tensor;
}

inline Tensor abs(const Tensor& tensor) {
    return Tensor(tensor.ctx_, ggml_abs(tensor.ctx_, tensor.t_));
}

inline Tensor sqrt(const Tensor& tensor) {
    return Tensor(tensor.ctx_, ggml_sqrt(tensor.ctx_, tensor.t_));
}

inline Tensor exp(const Tensor& tensor) {
    return Tensor(tensor.ctx_, ggml_exp(tensor.ctx_, tensor.t_));
}

inline Tensor log(const Tensor& tensor) {
    return Tensor(tensor.ctx_, ggml_log(tensor.ctx_, tensor.t_));
}

inline Tensor sin(const Tensor& tensor) {
    return Tensor(tensor.ctx_, ggml_sin(tensor.ctx_, tensor.t_));
}

inline Tensor cos(const Tensor& tensor) {
    return Tensor(tensor.ctx_, ggml_cos(tensor.ctx_, tensor.t_));
}

inline Tensor rsqrt(const Tensor& tensor) {
    return 1.0f / sqrt(tensor);
}

// TODO: implement:
/*template <size_t N>
Tensor reshape(std::array<int64_t, N> shape) const;

template <size_t N>
Tensor permute(std::array<int, N> order) const;

Tensor squeeze() const;

Tensor unsqueeze(int dim) const;

Tensor narrow(int dim, int64_t start, int64_t length) const;*/

inline std::vector<Tensor> Tensor::chunk(int n, int dim) const {
    if (dim < 0) dim += ndim();

    auto ne = this->shape();
    int64_t chunk_size = (ne[dim] + n - 1) / n;  // Ceiling division

    std::vector<Tensor> chunks;
    chunks.reserve(n);

    for (int i = 0; i < n && static_cast<int64_t>(i) * chunk_size < ne[dim]; ++i) {
        int64_t start = static_cast<int64_t>(i) * chunk_size;
        int64_t length = std::min(chunk_size, ne[dim] - start);
        chunks.push_back(narrow(dim, start, length));
    }

    return chunks;
}

inline std::vector<Tensor> Tensor::split(int64_t split_size, int dim) const {
    if (dim < 0) dim += ndim();

    auto ne = this->shape();
    std::vector<Tensor> splits;

    for (int64_t start = 0; start < ne[dim]; start += split_size) {
        int64_t length = std::min(split_size, ne[dim] - start);
        splits.push_back(narrow(dim, start, length));
    }

    return splits;
}

inline Tensor Tensor::cat(const std::vector<Tensor>& tensors, int dim) {
    if (tensors.size() == 1)
        return tensors.at(0);

    auto tensor = tensors.at(0);

    for (size_t i = 1; i < tensors.size(); ++i) {
        tensor = Tensor(tensor.ctx_, ggml_concat(tensor.ctx_, *tensor, *tensors.at(i), dim));
    }

    return tensor;
}
