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
 * Core tensor abstraction for the diffusers pipeline implementation in C++/.
 * Provides a PyTorch-like API that constructs ggml computation graph nodes without
 * executing computation. Actual execution happens via the backend scheduler (Scheduler).
 */

/** @brief Non-owning handle to a ggml_tensor within a ggml_context.
 *
 * - **Non-owning**: Tensor does not own its ggml_context or storage. This avoids per-tensor
 *   allocations and matches ggml's single-context model. In diffusers pipelines, forward passes
 *   construct long chains of tensor operations — materializing each as owned objects would add
 *   significant overhead. Tensors are valid only while the parent context remains alive.
 * - **Graph-building semantics**: All operations construct ggml graph nodes in-place; no computation
 *   occurs at construction time. The caller builds and executes the complete compute graph via
 *   Scheduler, matching PyTorch's lazy autograd but delegating execution to ggml.
 * - **View vs copy semantics**: `narrow` and `permute` produce view tensors that share the underlying
 *   buffer. `contiguous` always produces a copy tensor. Shape-changing operations require contiguous
 *   storage and rely on linear element counts rather than stride information.
 * - **Max rank of 4**: ggml tensors have a maximum rank of 4; Tensor::Shape enforces this with exactly
 *   4 dimension slots plus a logical size field indicating the actual number of dimensions used.
 */
class Tensor {
public:
    template<typename T>
    struct DType;

    /** @brief Shape descriptor for tensors with up to 4 dimensions.
     *
     * Stores exactly 4 dimension slots (matching ggml's maximum tensor rank) plus a logical size field
     * indicating how many dimensions are actually in use. The size field allows representing tensors of
     * any rank from 0 to 4 within the fixed 4-element array. Negative indexing is supported for operator[]
     * access (e.g., -1 refers to the last dimension).
     * 
     * GGML defines tensor dimensions as (ne0, ne1, ne2, ne3) where ne0 is the fastest-varying (innermost) dimension,
     * while PyTorch displays shapes in outermost → innermost order (left → right). From that, the “reverse ordering”
     * is a derived consequence, not a formally stated rule.
     */
    class Shape {
    public:
        static Shape broadcast(const Shape& lhs, const Shape& rhs);

        /** @brief Default-constructs a zero-shape with specified rank. */
        Shape(int64_t rank = 0) : ne_({0, 0, 0, 0}), rank_(rank) {}

        /** @brief Constructs a shape from an initializer list of dimension sizes.
         *
         * The number of elements in the list determines size. All remaining dimension slots are zeroed.
         */
        Shape(const std::initializer_list<int64_t>& list) : ne_({0, 0, 0, 0}), rank_(list.size()) {
            size_t i = 0;

            for (auto it = std::rbegin(list); it != std::rend(list); ++it)
                ne_[i++] = *it;
        }

        /** @brief Accesses the dimension at the given index. */
        const int64_t& operator [](const int64_t& index) const { return ne_[rank_ - 1 - normalize_index(index)]; }

        /** @brief Mutable access to the dimension at the given index. */
        int64_t& operator [](const int64_t& index) { return ne_[rank_ - 1 - normalize_index(index)]; }

        /** @brief Returns the logical number of dimensions (0–4). */
        const int64_t& rank() const { return rank_; }

        /** @brief Returns a pointer to the raw dimension array. */
        const int64_t* data() const { return ne_.data(); }

        /** @brief Converts the shape to a human-readable string representation. */
        std::string to_string() const;

        bool operator ==(const Shape& other) const {
            return ne_ == other.ne_ && rank_ == other.rank_;
        }

        bool operator !=(const Shape& other) const {
            return !(*this == other);
        }
    private:
        std::array<int64_t, 4> ne_;
        int64_t rank_;

        int64_t normalize_index(int64_t index) const {
            if (index < 0)
                index += rank_;

            if (index < 0 || index >= rank_)
                throw std::out_of_range("Shape index out of range: " + std::to_string(index) + ", but rank is " + std::to_string(rank_));

            return index;
        }

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

    /** @brief Default-constructs an invalid Tensor. */
    Tensor() : ctx_(nullptr), t_(nullptr) {}

    /** @brief Constructs a Tensor wrapping the given ggml pointers, inferring shape.
     *
     * This is the only way to create a valid Tensor from raw ggml objects. The Tensor does not assume ownership
     * of either pointer — it is valid only while both ctx and t remain alive.
     * 
     * @remarks When Tensor shape is inferred, GGML collapses single dimensions so result can be not
     * what is expected. It is highly recommended to always pass the logical shape when constructing a Tensor.
     */
    explicit Tensor(ggml_context* ctx, ggml_tensor* t)
        : ctx_(ctx), t_(t), shape_()
    {
        if (!ctx_ || !t_)
            return;

        Shape shape({t_->ne[3], t_->ne[2], t_->ne[1], t_->ne[0]});
        shape.rank_ = ggml_n_dims(t_); // NOTE: this collapses single dimensions
        shape_ = std::move(shape);
    }

    /** @brief Constructs a Tensor wrapping the given ggml pointers and a logical shape.
     *
     * This is the only way to create a valid Tensor from raw ggml objects. The Tensor does not assume ownership
     * of either pointer — it is valid only while both ctx and t remain alive.
     */
    explicit Tensor(ggml_context* ctx, ggml_tensor* t, const Shape& shape)
        : ctx_(ctx), t_(t), shape_(shape) {}

    /** @brief Returns the number of logical dimensions. */
    int ndim() const {
        return shape_.rank();
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

    /** @brief Returns the tensor context. */
    ggml_context* context() const {
        return ctx_;
    }

    /** @brief Returns true if tensor is valid; otherwise returns false. */
    operator bool() const {
        return ctx_ != nullptr && t_ != nullptr;
    }

    /** @brief Returns a contiguous copy of this tensor.
     *
     * Always produces a copy tensor regardless of whether the input is already contiguous. This is used when a 
     * shape-changing operation requires contiguous memory layout (e.g., reshape on non-contiguous tensors).
     */
    Tensor contiguous() const {
        throw_if_not_valid();
        return Tensor(ctx_, ggml_cont(ctx_, t_), shape_);
    }

    /** @brief Returns true if tensor is contiguous; otherwise returns false. */
    bool is_contiguous() const {
        throw_if_not_valid();
        return ggml_is_contiguous(t_);
    }

    /** @brief Returns the logical shape of this tensor. */
    const Shape& shape() const {
        return shape_;
    }

    /** @brief Returns the raw ggml_tensor pointer. */
    ggml_tensor* operator *() const {
        return t_;
    }

    /** @brief Returns a copy of this tensor. */
    Tensor clone() const {
        return Tensor(ctx_, ggml_dup(ctx_, t_), shape_);
    }
    
    /** @brief Returns a contiguous copy of this tensor. */
    Tensor clone_as_contiguous() const {
        if (!is_contiguous())
            return contiguous();

        return clone();
    }

    /** @brief Creates an uninitialized tensor with the given shape and type. */
    // TODO: remove this and move to runtime
    template <typename T> static Tensor empty(
        ggml_context* ctx,
        const Shape& shape)
    {
        return empty(ctx, shape, DType<T>::value);
    }

    /** @brief Creates a scalar tensor filled with the given value. */
    // TODO: remove this and move to runtime
    template <typename T> static Tensor scalar(ggml_context* ctx, const T& value) {
        // GGML supports filling only float tensors
        auto tensor = empty<float>(ctx, {});
        tensor.t_ = ggml_fill_inplace(tensor.ctx_, tensor.t_, (float)value);

        return tensor.to(DType<T>::value);
    }

    /** @brief Creates an uninitialized tensor with the given shape and type. */
    // TODO: remove this and move to runtime
    static Tensor empty(
        ggml_context* ctx,
        const Shape& shape,
        ggml_type type)
    {
        // GGML does not support 0d tensors, let's fake it with 1d tensor
        if (shape.rank() == 0)
            return Tensor(ctx, ggml_new_tensor_1d(ctx, type, 1), shape);

        return Tensor(ctx, ggml_new_tensor(ctx, type, shape.rank(), shape.data()), shape);
    }

    /** @brief Creates a filled tensor with the given value, shape and type. */
    // TODO: remove this and move to runtime
    static Tensor full(
        ggml_context* ctx,
        const Shape& shape,
        float value)
    {
        auto tensor = empty<float>(ctx, shape);
        tensor.t_ = ggml_fill_inplace(ctx, tensor.t_, value);
        
        return tensor;
    }

    /** @brief Creates a zero-filled tensor with the given shape and type. */
    // TODO: remove this and move to runtime
    static Tensor zeros(
        ggml_context* ctx,
        const Shape& shape)
    {
        return full(ctx, shape, 0.0f);
    }

    /** @brief Creates a one-filled tensor with the given shape and type. */
    // TODO: remove this and move to runtime
    static Tensor ones(
        ggml_context* ctx,
        const Shape& shape)
    {
        return full(ctx, shape, 1.0f);
    }

    // TODO: remove this and move to runtime
    static Tensor arange(
        ggml_context* ctx,
        float start,
        float stop,
        float step = 1.0f)
    {
        const int64_t size = static_cast<int64_t>(std::ceil((stop - start) / step));

        return Tensor(ctx, ggml_arange(ctx, start, stop, step), {size});
    }

    /** @brief Concatenates a list of tensors along the given dimension `dim`. */
    static Tensor cat(const std::vector<Tensor>& tensors, int dim);

    /** @brief Stacks a list of tensors along the given dimension `dim`. */
    static Tensor stack(const std::vector<Tensor>& tensors, int64_t dim);

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
    Tensor permute(const Shape& dims) const;

    /** @brief Swaps two dimensions, producing a view tensor. */
    Tensor transpose(int64_t dim0, int64_t dim1) const;

    /** @brief Removes a single dimension at the given index `dim` (must have size 1). */
    Tensor squeeze(int64_t dim) const;

    /** @brief Inserts a new dimension of size 1 at the given index `dim`. */
    Tensor unsqueeze(int64_t dim) const;

    /** @brief Flattens dimensions from `start_dim` to `end_dim` into a single dimension. */
    Tensor flatten(int64_t start_dim=0, int64_t end_dim=-1) const;

    /** @brief Reverses a previous flatten: splits dimension `dim` into the given shape. */
    Tensor unflatten(int64_t dim, const Shape& shape);

    /** @brief Narrows a dimension: keeps elements [start, start+length) along `dim`.
     *
     * This produces a view tensor (shares underlying buffer). Equivalent to PyTorch's narrow() indexing.
     */
    Tensor narrow(int64_t dim, int64_t start, int64_t length) const;

    Tensor expand(const Shape& new_shape) const;

    Tensor repeat(const Shape& repeats) const;

    /** @brief Chunks a tensor into `n` roughly equal pieces along dimension `dim`. */
    std::vector<Tensor> chunk(int n, int64_t dim = 0) const;

    /** @brief Splits a tensor into chunks of size `split_size` along dimension `dim`. */
    std::vector<Tensor> split(int64_t split_size, int64_t dim = 0) const;

    /** @brief Splits a tensor into chunks of specified sizes along dimension `dim`. */
    std::vector<Tensor> split_with_sizes(const std::vector<int64_t>& split_sizes, int64_t dim = 0) const;

    /** @brief Makes sure a tensor is type `type` if it is not already. */
    Tensor to(ggml_type type) const {
        if (dtype() != type)
            return cast(type);

        return *this;
    }

    /** @brief Converts a tensor to type `type`. */
    Tensor cast(ggml_type type) const {
        return Tensor(ctx_, ggml_cast(ctx_, t_, type), shape_);
    }

    Tensor operator -() const {
        return Tensor(ctx_, ggml_neg(ctx_, t_), shape_);
    }

    Tensor operator+(Tensor rhs) const {
        auto lhs = *this;
        auto target = Shape::broadcast(lhs.shape_, rhs.shape_);
        auto dtype = common_dtype(lhs.dtype(), rhs.dtype());

        lhs = lhs.to(dtype);
        rhs = rhs.to(dtype);

        lhs = lhs.expand(target);
        rhs = rhs.expand(target);

        return Tensor(lhs.ctx_, ggml_add(lhs.ctx_, lhs.t_, rhs.t_), lhs.shape_);
    }

    Tensor operator-(Tensor rhs) const {
        auto lhs = *this;
        auto target = Shape::broadcast(lhs.shape_, rhs.shape_);
        auto dtype = common_dtype(lhs.dtype(), rhs.dtype());

        lhs = lhs.to(dtype);
        rhs = rhs.to(dtype);

        lhs = lhs.expand(target);
        rhs = rhs.expand(target);

        return Tensor(lhs.ctx_, ggml_sub(ctx_, lhs.t_, rhs.t_), lhs.shape_).to(dtype);
    }

    Tensor operator*(Tensor rhs) const {
        auto lhs = *this;
        auto target = Shape::broadcast(lhs.shape_, rhs.shape_);
        auto dtype = common_dtype(lhs.dtype(), rhs.dtype());

        lhs = lhs.to(dtype);
        rhs = rhs.to(dtype);

        lhs = lhs.expand(target);
        rhs = rhs.expand(target);

        return Tensor(lhs.ctx_, ggml_mul(lhs.ctx_, lhs.t_, rhs.t_), lhs.shape_).to(dtype);
    }

    Tensor operator/(Tensor rhs) const {
        auto lhs = *this;
        auto target = Shape::broadcast(lhs.shape_, rhs.shape_);
        auto dtype = common_dtype(lhs.dtype(), rhs.dtype());

        lhs = lhs.to(dtype);
        rhs = rhs.to(dtype);

        lhs = lhs.expand(target);
        rhs = rhs.expand(target);

        return Tensor(lhs.ctx_, ggml_div(lhs.ctx_, lhs.t_, rhs.t_), lhs.shape_).to(dtype);
    }

    template <typename T>
    Tensor operator+(const T& rhs) const {
        return *this + scalar<T>(ctx_, rhs);
    }

    template <typename T>
    Tensor operator-(const T& rhs) const {
        return *this - scalar<T>(ctx_, rhs);
    }

    template <typename T>
    Tensor operator*(const T& rhs) const {
        return Tensor(ctx_, ggml_scale(ctx_, t_, (float)rhs), shape_);
    }

    template <typename T>
    Tensor operator/(const T& rhs) const {
        return Tensor(ctx_, ggml_scale(ctx_, t_, 1.0f / (float)rhs), shape_);
    }

    Tensor clamp(float a, float b) const {
        return Tensor(ctx_, ggml_clamp(ctx_, t_, a, b), shape_);
    }

    Tensor sum(int64_t dim, bool keepdim = false) const;

    Tensor mean(int64_t dim, bool keepdim = false) const;

    Tensor operator[](const size_t& index) const {
        return narrow(0, index, 1).squeeze(0);
    }

    Tensor operator [](const std::vector<Slice>& indices) const;
private:
    ggml_context* ctx_;
    ggml_tensor* t_;
    Shape shape_;

    template <typename T> friend Tensor operator-(const T& value, const Tensor& tensor);
    template <typename T> friend Tensor operator/(const T& value, const Tensor& tensor);
    friend Tensor abs(const Tensor& tensor);
    friend Tensor sqrt(const Tensor& tensor);
    friend Tensor exp(const Tensor& tensor);
    friend Tensor log(const Tensor& tensor);
    friend Tensor sin(const Tensor& tensor);
    friend Tensor cos(const Tensor& tensor);
    friend Tensor pow(const Tensor& base, const Tensor& exponent);
    friend Tensor pow(const Tensor& base, float exponent);
    friend Tensor pow(float base, const Tensor& exponent);

    static int normalize_dim(const std::string& method, int64_t dim, int64_t rank, bool allow_end = false);
    static ggml_type common_dtype(ggml_type lhs, ggml_type rhs);

    void throw_if_not_valid() const;
};

template <typename T>
inline Tensor operator+(const T& value, const Tensor& tensor) {
    return tensor + value;
}

template <typename T>
inline Tensor operator-(const T& value, const Tensor& tensor) {
    return Tensor::scalar<T>(tensor.ctx_, value) - tensor;
}

template <typename T>
inline Tensor operator*(const T& value, const Tensor& tensor) {
    return tensor * value;
}

template <typename T>
inline Tensor operator/(const T& value, const Tensor& tensor) {
    return Tensor::scalar<T>(tensor.ctx_, value) / tensor;
}

inline Tensor abs(const Tensor& tensor) {
    return Tensor(tensor.ctx_, ggml_abs(tensor.ctx_, tensor.t_), tensor.shape_);
}

inline Tensor sqrt(const Tensor& tensor) {
    return Tensor(tensor.ctx_, ggml_sqrt(tensor.ctx_, tensor.t_), tensor.shape_);
}

inline Tensor exp(const Tensor& tensor) {
    return Tensor(tensor.ctx_, ggml_exp(tensor.ctx_, tensor.t_), tensor.shape_);
}

inline Tensor log(const Tensor& tensor) {
    return Tensor(tensor.ctx_, ggml_log(tensor.ctx_, tensor.t_), tensor.shape_);
}

inline Tensor sin(const Tensor& tensor) {
    return Tensor(tensor.ctx_, ggml_sin(tensor.ctx_, tensor.t_), tensor.shape_);
}

inline Tensor cos(const Tensor& tensor) {
    return Tensor(tensor.ctx_, ggml_cos(tensor.ctx_, tensor.t_), tensor.shape_);
}

inline Tensor pow(const Tensor& base, const Tensor& exponent) {
    // exp(exponent * log(x))  — works for arbitrary real exponents
    return exp(log(base) * exponent);
}

inline Tensor pow(const Tensor& base, float exponent) {
    return pow(base, Tensor::scalar(base.ctx_, exponent));
}

inline Tensor pow(float base, const Tensor& exponent) {
    return pow(Tensor::scalar(exponent.ctx_, base), exponent);
}

inline Tensor rsqrt(const Tensor& tensor) {
    return 1.0f / sqrt(tensor);
}

template<>
struct Tensor::DType<float> {
    static constexpr ggml_type value = GGML_TYPE_F32;
};

template<>
struct Tensor::DType<int64_t> {
    static constexpr ggml_type value = GGML_TYPE_I64;
};

template<>
struct Tensor::DType<int32_t> {
    static constexpr ggml_type value = GGML_TYPE_I32;
};

template<>
struct Tensor::DType<int16_t> {
    static constexpr ggml_type value = GGML_TYPE_I16;
};

template<>
struct Tensor::DType<int8_t> {
    static constexpr ggml_type value = GGML_TYPE_I8;
};
