#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <optional>

#include "ggml.h"

/** @file Tensor.hpp
 *
 * Core tensor abstraction for the diffusers pipeline implementation in C++/GGML.
 * Provides a PyTorch-like API that constructs ggml computation graph nodes without
 * executing computation. Actual execution happens via the backend scheduler (GGMLScheduler).
 */

/** @brief Non-owning handle to a ggml_tensor within a ggml_context.
 *
 * - **Non-owning**: Tensor does not own its ggml_context or storage. This avoids per-tensor
 *   allocations and matches ggml's single-context model. In diffusers pipelines, forward passes
 *   construct long chains of tensor operations — materializing each as owned objects would add
 *   significant overhead. Tensors are valid only while the parent context remains alive.
 * - **Graph-building semantics**: All operations construct ggml graph nodes in-place; no computation
 *   occurs at construction time. The caller builds and executes the complete compute graph via
 *   GGMLScheduler, matching PyTorch's lazy autograd but delegating execution to ggml.
 * - **View vs copy semantics**: `narrow` and `permute` produce view tensors that share the underlying
 *   buffer. `contiguous` always produces a copy tensor. Shape-changing operations require contiguous
 *   storage and rely on linear element counts rather than stride information.
 * - **Max rank of 4**: ggml tensors have a maximum rank of 4; Tensor::Shape enforces this with exactly
 *   4 dimension slots plus a logical size field indicating the actual number of dimensions used.
 */
class Tensor {
public:
    /** @brief Shape descriptor for tensors with up to 4 dimensions.
     *
     * Stores exactly 4 dimension slots (matching ggml's maximum tensor rank) plus a logical size field
     * indicating how many dimensions are actually in use. The size field allows representing tensors of
     * any rank from 0 to 4 within the fixed 4-element array. Negative indexing is supported for operator[]
     * access (e.g., -1 refers to the last dimension).
     */
    class Shape {
    public:
        using iterator = std::array<int64_t, 4>::iterator;
        using const_iterator = std::array<int64_t, 4>::const_iterator;

        /** @brief Default-constructs a zero-shape with specified rank. */
        Shape(size_t rank = 0) : ne_({0, 0, 0, 0}), rank_(rank) {}

        /** @brief Constructs a shape from an initializer list of dimension sizes.
         *
         * The number of elements in the list determines size. All remaining dimension slots are zeroed.
         */
        Shape(const std::initializer_list<int64_t>& list) : ne_({0, 0, 0, 0}), rank_(list.size()) {
            size_t i = 0;

            for (auto it = std::begin(list); it != std::end(list); ++it)
                ne_[i++] = *it;
        }

        /** @brief Accesses the dimension at the given index. */
        const int64_t& operator [](const size_t& index) const { return ne_[index]; }

        /** @brief Mutable access to the dimension at the given index. */
        int64_t& operator [](const size_t& index) { return ne_[index]; }

        /** @brief Returns the logical number of dimensions (0–4). */
        size_t rank() const { return rank_; }

        /** @brief Returns a pointer to the raw dimension array. */
        const int64_t* data() const { return ne_.data(); }

        /** @brief Converts the shape to a human-readable string representation. */
        std::string to_string() const;

        iterator begin() { return ne_.begin(); }
        iterator end() { return ne_.end(); }
        const_iterator begin() const { return ne_.begin(); }
        const_iterator end() const { return ne_.end(); }

    private:
        std::array<int64_t, 4> ne_;
        int rank_;

        friend class Tensor;
    };

    /** @brief Describes a dimension slice for indexing operations.
     *
     * Supports PyTorch-like slicing syntax: full slice (:), single index (i), range with step (start:stop:step),
     * new axis insertion (None), and ellipsis (...). Used by operator[] with a vector of Slices for advanced indexing.
     */
    struct Slice {
        /** @brief The type of slice operation to perform. */
        enum class Kind {
            All,        // : — select all elements along dimension
            Index,      // i — select single element at index i (dimension is removed)
            Range,      // start:stop:step — select elements in range with step
            NewAxis,    // None — insert a new dimension of size 1
            Ellipsis,   // ... — expand to fill missing dimensions
        };

        Kind kind = Kind::All;
        std::optional<int64_t> start;
        std::optional<int64_t> stop;
        int64_t step = 1;

        /** @brief Creates a full-slice (:). */
        static const Slice all() {
            return {};
        }

        /** @brief Creates an index slice (i) selecting element at position i. */
        static Slice index(int64_t i) {
            Slice s;
            s.kind = Kind::Index;
            s.start = i;
            return s;
        }

        /** @brief Creates a range slice (start:stop:step). */
        static Slice range(std::optional<int64_t> start, std::optional<int64_t> stop, int64_t step = 1) {
            Slice s;
            s.kind = Kind::Range;
            s.start = start;
            s.stop = stop;
            s.step = step;
            return s;
        }

        /** @brief Creates a new-axis slice (None). */
        static Slice none() {
            Slice s;
            s.kind = Kind::NewAxis;
            return s;
        }

        /** @brief Creates an ellipsis slice (...). */
        static Slice ellipsis() {
            Slice s;
            s.kind = Kind::Ellipsis;
            return s;
        }
    };

    /** @brief Default-constructs an invalid (empty) Tensor. */
    Tensor() : ctx_(nullptr), t_(nullptr) {}

    /** @brief Constructs a Tensor wrapping the given ggml pointers.
     *
     * This is the only way to create a valid Tensor from raw ggml objects. The Tensor does not assume ownership
     * of either pointer — it is valid only while both ctx and t remain alive.
     */
    explicit Tensor(ggml_context* ctx, ggml_tensor* t)
        : ctx_(ctx), t_(t) {}

    /** @brief Returns the number of logical dimensions (0–4). */
    int ndim() const {
        if (!ctx_ || !t_)
            return 0;

        return ggml_n_dims(t_);
    }

    /** @brief Returns the total number of elements in the tensor. */
    int64_t numel() const {
        if (!ctx_ || !t_)
            return 0;

        return ggml_nelements(t_);
    }

    /** @brief Returns the data type of the underlying ggml_tensor. */
    ggml_type dtype() const {
        throw_if_not_valid();
        return t_->type;
    }

    /** @brief Returns a contiguous copy of this tensor.
     *
     * Always produces a copy tensor regardless of whether the input is already contiguous. This is used when a 
     * shape-changing operation requires contiguous memory layout (e.g., reshape on non-contiguous tensors).
     */
    Tensor contiguous() const {
        throw_if_not_valid();
        return Tensor(ctx_, ggml_cont(ctx_, t_));
    }

    /** @brief Returns the logical shape of this tensor. */
    Shape shape() const {
        if (!ctx_ || !t_)
            return Shape();

        Shape shape({t_->ne[0], t_->ne[1], t_->ne[2], t_->ne[3]});
        shape.rank_ = ndim();
        return std::move(shape);
    }

    /** @brief Returns the raw ggml_tensor pointer. */
    ggml_tensor* operator *() const {
        return t_;
    }


    /** @brief Creates an uninitialized tensor with the given shape and type. */
    static Tensor empty(
        ggml_context* ctx,
        ggml_type type,
        const Shape& shape)
    {
        return Tensor(ctx, ggml_new_tensor(ctx, type, shape.rank(), shape.data()));
    }

    /** @brief Creates a scalar tensor filled with the given value. */
    static Tensor scalar(
        ggml_context* ctx,
        float value,
        ggml_type type = GGML_TYPE_F32)
    {
        auto tensor = empty(ctx, type, {1});
        tensor.t_ = ggml_fill_inplace(tensor.ctx_, tensor.t_, value);

        return tensor;
    }

    /** @brief Creates a zero-filled tensor with the given shape and type. */
    static Tensor zeros(
        ggml_context* ctx,
        const Shape& shape,
        ggml_type type = GGML_TYPE_F32)
    {
        auto tensor = empty(ctx, type, shape);
        tensor.t_ = ggml_fill_inplace(ctx, tensor.t_, 0.0f);
        return tensor;
    }

    /** @brief Creates a one-filled tensor with the given shape and type. */
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

    /** @brief Concatenates a list of tensors along the given dimension `dim`. */
    static Tensor cat(const std::vector<Tensor>& tensors, int dim);


    /** @brief Reshapes the tensor to a new shape.
     *
     * If the current tensor is non-contiguous, it is first made contiguous (copy). Shape-changing operations 
     * rely on linear element counts — the total number of elements must match. A dimension value of -1 
     * triggers automatic inference from the remaining dimensions (total elements / product of known dims).
     */
    Tensor reshape(const Shape& shape) const;

    /** @brief Permutes all four dimensions according to `order` [d0, d1, d2, d3].
     *
     * This produces a view tensor (shares underlying buffer). Negative indices are supported for each 
     * dimension in the order (e.g., -1 refers to the last dimension). Dimensions beyond ndim() are clamped.
     */
    Tensor permute(const Shape& order) const;

    /** @brief Removes a single dimension at the given index `dim` (must have size 1). */
    Tensor squeeze(int dim) const;

    /** @brief Inserts a new dimension of size 1 at the given index `dim`. */
    Tensor unsqueeze(int dim) const;

    /** @brief Flattens dimensions from `start_dim` to `end_dim` into a single dimension. */
    Tensor flatten(int start_dim, int end_dim) const;

    /** @brief Reverses a previous flatten: splits dimension `dim` into the given shape. */
    Tensor unflatten(int64_t dim, const Shape& shape);

    /** @brief Narrows a dimension: keeps elements [start, start+length) along `dim`.
     *
     * This produces a view tensor (shares underlying buffer). Equivalent to PyTorch's narrow() indexing.
     */
    Tensor narrow(int dim, int64_t start, int64_t length) const;

    Tensor expand(const Shape& new_shape) const;

    /** @brief Chunks a tensor into `n` roughly equal pieces along dimension `dim`. */
    std::vector<Tensor> chunk(int n, int dim = 0) const;

    /** @brief Splits a tensor into chunks of size `split_size` along dimension `dim`. */
    std::vector<Tensor> split(int64_t split_size, int dim = 0) const;

    /** @brief Splits a tensor into chunks of specified sizes along dimension `dim`. */
    std::vector<Tensor> split_with_sizes(const std::vector<int64_t>& split_sizes, int dim) const;

    /** @brief Casts a tensor to type `type`. */
    Tensor to(ggml_type type) const {
        if (type == dtype())
            return *this;
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

    Tensor operator+(float rhs) const {
        return *this + scalar(ctx_, rhs);
    }

    Tensor operator-(float rhs) const {
        return *this - scalar(ctx_, rhs);
    }

    Tensor operator*(float rhs) const {
        return Tensor(ctx_, ggml_scale(ctx_, t_, rhs));
    }

    Tensor operator/(float rhs) const {
        return *this / scalar(ctx_, rhs);
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

    /** @brief Computes the mean along a specific dimension.
     *
     * - When `dim` is omitted (default -1), sums all elements and divides by total count (reduces to scalar).
     * - When `dim` is specified, computes mean along that dimension only. With `keepdims=true`, the reduced
     *   dimension is preserved as size 1 for correct broadcasting. With `keepdims=false`, the dimension is removed.
     *
     * Implementation uses ggml_sum_rows for axis 1 (most common case in transformer models). For the last
     * dimension of rank-3 tensors, reshapes to [N, D], sums rows, then reshapes back. General cases fall back
     * to a reshape-sum-divide-broadcast approach.
     */
    Tensor mean(int dim = -1, bool keepdims = true) const {
        const int rank = ndim();
        dim = normalize_dim(dim, rank);

        // No dimension specified (dim == -1 before normalization): reduce all elements to scalar.
        if (dim < 0)
            return Tensor(ctx_, ggml_mean(ctx_, t_));

        // Axis 1: use ggml_sum_rows directly — most common case in transformers.
        if (dim == 1 && rank >= 2) {
            auto sum = Tensor(ctx_, ggml_sum_rows(ctx_, t_));
            auto divisor = Tensor::scalar(ctx_, static_cast<float>(t_->ne[1]), dtype());
            auto result = sum / divisor;

            if (!keepdims)
                return result.squeeze(1);

            return result;
        }

        // Last dimension of rank-3 tensor [N, D] → reshape to [N, D], sum rows → [N, 1].
        if (dim == 2 && rank == 3) {
            int64_t N = t_->ne[0] * t_->ne[1];
            int64_t D = t_->ne[2];

            auto reshaped = Tensor(ctx_, ggml_reshape_2d(ctx_, t_, N, D));
            auto sum = Tensor(ctx_, ggml_sum_rows(ctx_, reshaped.t_));
            auto divisor = Tensor::scalar(ctx_, static_cast<float>(D), dtype());
            auto result = sum / divisor; // [N, 1]

            if (!keepdims)
                return result.squeeze(1);

            // Reshape back to [*, *, 1].
            std::array<int64_t, 4> ne = {t_->ne[0], t_->ne[1], 1, 1};
            return Tensor(ctx_, ggml_reshape_3d(ctx_, result.t_, ne[0], ne[1], ne[2]));
        }

        // General case: reshape so target dim becomes axis 1, sum_rows, then reshape back.
        {
            std::array<int64_t, 4> ne = {t_->ne[0], t_->ne[dim], 1, 1};
            int64_t N = 1;
            for (int i = 0; i < dim; ++i) N *= t_->ne[i];

            auto reshaped = Tensor(ctx_, ggml_reshape_2d(ctx_, t_, N, ne[1]));
            auto sum = Tensor(ctx_, ggml_sum_rows(ctx_, reshaped.t_));
            auto divisor = Tensor::scalar(ctx_, static_cast<float>(ne[1]), dtype());
            auto result = sum / divisor; // [N, 1]

            if (!keepdims) {
                return result.squeeze(1);
            }

            // Reconstruct shape with target dim = 1.
            std::array<int64_t, 4> out_ne = {1, 1, 1, 1};
            int out_rank = rank;
            for (int i = 0; i < dim; ++i)       out_ne[i] = t_->ne[i];
            out_ne[dim] = 1;
            for (int i = dim + 1; i < rank; ++i) out_ne[i] = t_->ne[i];

            switch (out_rank) {
                case 2: return Tensor(ctx_, ggml_reshape_2d(ctx_, result.t_, out_ne[0], out_ne[1]));
                case 3: return Tensor(ctx_, ggml_reshape_3d(ctx_, result.t_, out_ne[0], out_ne[1], out_ne[2]));
                default: return Tensor(ctx_, ggml_reshape_4d(ctx_, result.t_, out_ne[0], out_ne[1], out_ne[2], out_ne[3]));
            }
        }
    }

    Tensor operator[](const size_t& index) const {
        return narrow(0, index, 1).squeeze(0);
    }

    Tensor operator [](const std::vector<Slice>& indices) const;
    
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
