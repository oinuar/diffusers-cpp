#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <limits>

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

    std::array<int64_t, 4> shape() const {
        // TODO: make better shape type to avoid passing all the dims here
        return {t_->ne[0], t_->ne[1], t_->ne[2], t_->ne[3]};
    }

    ggml_tensor* operator *() const {
        return t_;
    }


    template <size_t N>
    static Tensor empty(
        ggml_context* ctx,
        ggml_type type,
        std::array<int64_t, N> shape)
    {
        static_assert(N >= 1 && N <= 4, "ggml supports tensor ranks from 1 to 4");
        return Tensor(ctx, ggml_new_tensor(ctx, type, N, shape.data()));
    }

    static Tensor scalar(
        ggml_context* ctx,
        float value,
        ggml_type type = GGML_TYPE_F32)
    {
        auto tensor = empty<1>(ctx, type, {1});
        tensor.t_ = ggml_fill_inplace(tensor.ctx_, tensor.t_, value);

        return tensor;
    }

    template <size_t N>
    static Tensor zeros(
        ggml_context* ctx,
        std::array<int64_t, N> shape,
        ggml_type type = GGML_TYPE_F32)
    {
        static_assert(N >= 1 && N <= 4, "ggml supports tensor ranks from 1 to 4");

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
        static_assert(N >= 1 && N <= 4, "ggml supports tensor ranks from 1 to 4");

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

template <size_t N>
Tensor Tensor::reshape(std::array<int64_t, N> new_shape) const {
    static_assert(N >= 1 && N <= 4, "ggml supports tensor ranks from 1 to 4");
    throw_if_not_valid();

    // A reshape preserves linear storage order. A permuted/view tensor may not
    // have that order, so require contiguous storage.
    throw_if_not_contiguous();

    std::array<int64_t, 4> ne = { 1, 1, 1, 1 };

    int infer_dim = -1;
    int64_t known_product = 1;

    for (size_t i = 0; i < N; ++i) {
        const int64_t d = new_shape[i];

        if (d == -1) {
            if (infer_dim != -1) {
                throw std::invalid_argument("reshape allows only one -1 dimension");
            }
            infer_dim = static_cast<int>(i);
            continue;
        }

        if (d <= 0) {
            throw std::invalid_argument(
                "reshape dimensions must be positive, except one -1");
        }

        if (known_product > std::numeric_limits<int64_t>::max() / d) {
            throw std::overflow_error("reshape shape is too large");
        }

        known_product *= d;
        ne[i] = d;
    }

    const int64_t old_numel = numel();

    if (infer_dim != -1) {
        if (known_product == 0 || old_numel % known_product != 0) {
            throw std::invalid_argument(
                "cannot infer reshape dimension: element counts do not match");
        }

        ne[infer_dim] = old_numel / known_product;
    }

    if (checked_numel(ne, static_cast<int>(N)) != old_numel) {
        throw std::invalid_argument(
            "reshape cannot change the number of elements");
    }

    switch (N) {
        case 1:
            return Tensor(ctx_, ggml_reshape_1d(ctx_, t_, ne[0]));
        case 2:
            return Tensor(ctx_, ggml_reshape_2d(ctx_, t_, ne[0], ne[1]));
        case 3:
            return Tensor(ctx_, ggml_reshape_3d(ctx_, t_, ne[0], ne[1], ne[2]));
        case 4:
            return Tensor(ctx_, ggml_reshape_4d(
                ctx_, t_, ne[0], ne[1], ne[2], ne[3]));
        default:
            throw std::logic_error("unreachable");
    }
}


template <size_t N>
Tensor Tensor::permute(std::array<int, N> order) const {
    static_assert(N >= 1 && N <= 4, "ggml supports tensor ranks from 1 to 4");
    throw_if_not_valid();

    const int rank = ndim();

    // Permit a full 4-axis ggml permutation, or a permutation of only the
    // logical dimensions. The latter leaves trailing singleton axes unchanged.
    if (static_cast<int>(N) != rank && N != 4) {
        throw std::invalid_argument(
            "permute order must have ndim() entries or exactly 4 entries");
    }

    std::array<int, 4> axes = { 0, 1, 2, 3 };
    std::array<bool, 4> seen = { false, false, false, false };

    for (size_t i = 0; i < N; ++i) {
        int axis = order[i];

        if (axis < 0) {
            axis += rank;
        }

        if (axis < 0 || axis >= rank) {
            throw std::out_of_range("permute axis out of range");
        }

        if (seen[axis]) {
            throw std::invalid_argument("permute order contains duplicate axes");
        }

        seen[axis] = true;
        axes[i] = axis;
    }

    // If only logical axes were supplied, all logical axes must occur once.
    // Trailing physical ggml singleton axes remain identity-mapped.
    if (N != 4) {
        for (int axis = 0; axis < rank; ++axis) {
            if (!seen[axis]) {
                throw std::invalid_argument(
                    "permute order must contain every tensor dimension");
            }
        }
    } else {
        // For a 4-entry order, remaining physical axes must also be represented.
        for (int axis = 0; axis < 4; ++axis) {
            if (!seen[axis]) {
                throw std::invalid_argument(
                    "4D permute order must contain axes 0, 1, 2, and 3");
            }
        }
    }

    return Tensor(ctx_, ggml_permute(
        ctx_, t_, axes[0], axes[1], axes[2], axes[3]));
}


inline Tensor Tensor::squeeze() const {
    throw_if_not_valid();
    throw_if_not_contiguous();

    const int rank = ndim();
    const auto old_ne = shape();

    std::array<int64_t, 4> new_ne = { 1, 1, 1, 1 };
    int new_rank = 0;

    for (int i = 0; i < rank; ++i) {
        if (old_ne[i] != 1) {
            new_ne[new_rank++] = old_ne[i];
        }
    }

    // ggml has no rank-0 tensor representation. Match a practical tensor API:
    // squeezing an all-singleton tensor leaves a rank-1 [1] tensor.
    if (new_rank == 0) {
        new_rank = 1;
        new_ne[0] = 1;
    }

    switch (new_rank) {
        case 1:
            return Tensor(ctx_, ggml_reshape_1d(ctx_, t_, new_ne[0]));
        case 2:
            return Tensor(ctx_, ggml_reshape_2d(
                ctx_, t_, new_ne[0], new_ne[1]));
        case 3:
            return Tensor(ctx_, ggml_reshape_3d(
                ctx_, t_, new_ne[0], new_ne[1], new_ne[2]));
        case 4:
            return Tensor(ctx_, ggml_reshape_4d(
                ctx_, t_, new_ne[0], new_ne[1], new_ne[2], new_ne[3]));
        default:
            throw std::logic_error("ggml tensor rank must be between 1 and 4");
    }
}


inline Tensor Tensor::unsqueeze(int dim) const {
    throw_if_not_valid();
    throw_if_not_contiguous();

    const int old_rank = ndim();

    if (old_rank >= 4) {
        throw std::invalid_argument(
            "cannot unsqueeze: ggml supports at most 4 dimensions");
    }

    dim = normalize_dim(dim, old_rank, true);

    const auto old_ne = shape();
    std::array<int64_t, 4> new_ne = { 1, 1, 1, 1 };

    for (int src = 0, dst = 0; src < old_rank; ++src, ++dst) {
        if (dst == dim) {
            ++dst;
        }

        new_ne[dst] = old_ne[src];
    }

    const int new_rank = old_rank + 1;

    switch (new_rank) {
        case 2:
            return Tensor(ctx_, ggml_reshape_2d(
                ctx_, t_, new_ne[0], new_ne[1]));
        case 3:
            return Tensor(ctx_, ggml_reshape_3d(
                ctx_, t_, new_ne[0], new_ne[1], new_ne[2]));
        case 4:
            return Tensor(ctx_, ggml_reshape_4d(
                ctx_, t_, new_ne[0], new_ne[1], new_ne[2], new_ne[3]));
        default:
            throw std::logic_error("ggml tensor rank must be between 1 and 4");
    }
}


inline Tensor Tensor::narrow(int dim, int64_t start, int64_t length) const {
    throw_if_not_valid();

    const int rank = ndim();
    dim = normalize_dim(dim, rank);

    if (start < 0 || length < 0) {
        throw std::invalid_argument("narrow start and length must be non-negative");
    }

    const auto ne = shape();

    if (start > ne[dim] || length > ne[dim] - start) {
        throw std::out_of_range("narrow range exceeds tensor dimension");
    }

    std::array<int64_t, 4> view_ne = ne;
    view_ne[dim] = length;

    // ggml strides (nb) are byte strides. Moving start elements along dim
    // therefore advances start * nb[dim] bytes.
    const size_t offset =
        static_cast<size_t>(start) * static_cast<size_t>(t_->nb[dim]);

    switch (rank) {
        case 1:
            return Tensor(ctx_, ggml_view_1d(
                ctx_, t_, view_ne[0], offset));

        case 2:
            return Tensor(ctx_, ggml_view_2d(
                ctx_, t_,
                view_ne[0], view_ne[1],
                t_->nb[1],
                offset));

        case 3:
            return Tensor(ctx_, ggml_view_3d(
                ctx_, t_,
                view_ne[0], view_ne[1], view_ne[2],
                t_->nb[1], t_->nb[2],
                offset));

        case 4:
            return Tensor(ctx_, ggml_view_4d(
                ctx_, t_,
                view_ne[0], view_ne[1], view_ne[2], view_ne[3],
                t_->nb[1], t_->nb[2], t_->nb[3],
                offset));

        default:
            throw std::logic_error("ggml tensor rank must be between 1 and 4");
    }
}

inline std::vector<Tensor> Tensor::chunk(int n, int dim) const {
    throw_if_not_valid();

    if (n == 0)
        throw std::invalid_argument("n should be > 0");

    dim = normalize_dim(dim, ndim());

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
    if (split_size == 0)
        throw std::invalid_argument("split_size should be > 0");

    if (dim < 0)
        dim += ndim();

    auto ne = this->shape();
    std::vector<Tensor> splits;

    for (int64_t start = 0; start < ne[dim]; start += split_size) {
        int64_t length = std::min(split_size, ne[dim] - start);
        splits.push_back(narrow(dim, start, length));
    }

    return splits;
}

Tensor Tensor::cat(const std::vector<Tensor>& tensors, int dim) {
    if (tensors.empty())
        throw std::invalid_argument("cat requires at least one tensor");

    const Tensor& first = tensors.front();
    first.throw_if_not_valid();

    const int rank = first.ndim();
    dim = normalize_dim(dim, rank);

    for (const auto& x : tensors) {
        x.throw_if_not_valid();

        if (x.ndim() != rank)
            throw std::invalid_argument("all tensors passed to cat must have the same rank");

        if (x.dtype() != first.dtype())
            throw std::invalid_argument("all tensors passed to cat must have the same dtype");

        const auto a = first.shape();
        const auto b = x.shape();

        for (int d = 0; d < rank; ++d) {
            if (d != dim && a[d] != b[d]) {
                throw std::invalid_argument(
                    "cat requires matching sizes in all non-concatenated dimensions");
            }
        }
    }

    Tensor result = first;
    for (size_t i = 1; i < tensors.size(); ++i) {
        result = Tensor(
            result.ctx_,
            ggml_concat(result.ctx_, result.t_, tensors[i].t_, dim));
    }

    return result;
}

inline int Tensor::normalize_dim(int dim, int rank, bool allow_end) {
    const int upper = allow_end ? rank : rank - 1;

    if (dim < 0) {
        dim += allow_end ? (rank + 1) : rank;
    }

    if (dim < 0 || dim > upper) {
        throw std::out_of_range("tensor dimension out of range");
    }

    return dim;
}

inline int64_t Tensor::checked_numel(const std::array<int64_t, 4>& shape, int rank) {
    int64_t result = 1;

    for (int i = 0; i < rank; ++i) {
        if (shape[i] <= 0) {
            throw std::invalid_argument("tensor dimensions must be positive");
        }

        // Avoid silent overflow in shape validation.
        if (result > INT64_MAX / shape[i]) {
            throw std::overflow_error("tensor shape is too large");
        }

        result *= shape[i];
    }

    return result;
}

inline void Tensor::throw_if_not_valid() const {
    if (!ctx_ || !t_)
        throw std::runtime_error("undefined Tensor");
}

inline void Tensor::throw_if_not_contiguous() const {
    if (!ggml_is_contiguous(t_))
        throw std::runtime_error(
            "a contiguous tensor is required; call contiguous() first");
}
