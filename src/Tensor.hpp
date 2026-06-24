#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <cmath>

#include "ggml.h"

/**
 * C++ wrapper around `ggml_tensor*` with a torch.Tensor-like API.
 *
 * Operations construct new ggml graph nodes in the same ggml_context.
 * No computation occurs until the caller builds and executes a ggml graph.
 *
 * The Tensor and all derived Tensor objects are valid only while ctx_ remains alive.
 * Tensor does not own the ggml_context or ggml_tensor storage.
 */
class Tensor {
public:
    struct Shape {
    public:
        Shape() : ne_({0, 0, 0, 0}), size_(0) {}
        Shape(const std::initializer_list<int64_t>& list) : ne_({0, 0, 0, 0}), size_(list.size()) {
            size_t i = 0;

            for (auto it = std::begin(list); it != std::end(list); ++it) 
                ne_[i++] = *it;
        }

        const int64_t& operator [](const size_t& index) const { return ne_[index]; }

        int64_t& operator [](const size_t& index) { return ne_[index]; }

        size_t size() const { return size_; }

        const int64_t* data() const { return ne_.data(); }

        std::string to_string() const;

    private:
        std::array<int64_t, 4> ne_;
        int size_;

        friend class Tensor;
    };

    Tensor() : ctx_(nullptr), t_(nullptr)
    {
    }

    explicit Tensor(ggml_context* ctx, ggml_tensor* t)
        : ctx_(ctx), t_(t) {}

    int ndim() const {
        if (!ctx_ || !t_)
            return 0;

        return ggml_n_dims(t_);
    }

    int64_t numel() const {
        if (!ctx_ || !t_)
            return 0;

        return ggml_nelements(t_);
    }

    ggml_type dtype() const {
        throw_if_not_valid();
        return t_->type;
    }

    Tensor contiguous() const {
        throw_if_not_valid();
        return Tensor(ctx_, ggml_cont(ctx_, t_));
    }

    Shape shape() const {
        if (!ctx_ || !t_)
            return Shape();

        Shape shape({t_->ne[0], t_->ne[1], t_->ne[2], t_->ne[3]});
        shape.size_ = ndim();
        return std::move(shape);
    }

    ggml_tensor* operator *() const {
        return t_;
    }


    static Tensor empty(
        ggml_context* ctx,
        ggml_type type,
        const Shape& shape)
    {
        return Tensor(ctx, ggml_new_tensor(ctx, type, shape.size(), shape.data()));
    }

    static Tensor scalar(
        ggml_context* ctx,
        float value,
        ggml_type type = GGML_TYPE_F32)
    {
        auto tensor = empty(ctx, type, {1});
        tensor.t_ = ggml_fill_inplace(tensor.ctx_, tensor.t_, value);

        return tensor;
    }

    static Tensor zeros(
        ggml_context* ctx,
        const Shape& shape,
        ggml_type type = GGML_TYPE_F32)
    {
        auto tensor = empty(ctx, type, shape);
        tensor.t_ = ggml_fill_inplace(ctx, tensor.t_, 0.0f);
        return tensor;
    }

    static Tensor ones(
        ggml_context* ctx,
        const Shape& shape,
        ggml_type type = GGML_TYPE_F32)
    {
        auto tensor = empty(ctx, type, shape);
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


    Tensor reshape(const Shape& shape) const;

    /// Permute all four dimensions according to `order` [d0, d1, d2, d3].
    /// Negative indices are supported. Dimensions beyond ndim() are clamped.
    Tensor permute(const Shape& order) const;

    Tensor squeeze() const;

    Tensor unsqueeze(int dim) const;

    Tensor flatten(const Shape& shape) const;

    Tensor unflatten(int64_t dim, const Shape& shape);

    /// Narrow a dimension: keep elements [start, start+length) along `dim`.
    Tensor narrow(int dim, int64_t start, int64_t length) const;

    /// Chunk a tensor into `n` roughly equal pieces along dimension `dim`.
    std::vector<Tensor> chunk(int n, int dim = 0) const;

    /// Split a tensor into chunks of size `split_size` along dimension `dim`.
    std::vector<Tensor> split(int64_t split_size, int dim = 0) const;

    Tensor to(ggml_type type) const {
        return Tensor(ctx_, ggml_cast(ctx_, t_, type));
    }

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

    Tensor pow(float exponent) const {
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
        return Tensor(ctx_, ggml_mean(ctx_, t_));
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

    static int normalize_dim(int dim, int rank, bool allow_end = false);
    static int64_t checked_numel(const std::array<int64_t, 4>& shape, int rank);

    void throw_if_not_valid() const;
    void throw_if_not_contiguous() const;
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
