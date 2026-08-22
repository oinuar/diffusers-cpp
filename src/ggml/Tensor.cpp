#include "ggml/Tensor.hpp"
#include <limits>
#include <sstream>
#include <algorithm>

std::string Tensor::Shape::to_string() const {
    std::ostringstream oss;
    oss << '(';

    for (auto i = 0; i < rank(); ++i) {
        if (i > 0)
            oss << ", ";
        oss << (*this)[i];
    }

    oss << ')';
    return oss.str();
}

Tensor::Shape Tensor::Shape::broadcast(const Tensor::Shape& lhs, const Tensor::Shape& rhs) {
    const size_t rank = std::max(lhs.rank(), rhs.rank());

    if (rank == 0)
        return Tensor::Shape();

    if (lhs.rank() == 0)
        return rhs;

    if (rhs.rank() == 0)
        return lhs;

    Shape result(rank);

    for (size_t i = 0; i < rank; ++i) {
        const int64_t dl = (i < lhs.rank()) ? lhs.ne_[i] : 1;
        const int64_t dr = (i < rhs.rank()) ? rhs.ne_[i] : 1;

        if (dl == dr) {
            result.ne_[i] = dl;
        } else if (dl == 1) {
            result.ne_[i] = dr;
        } else if (dr == 1) {
            result.ne_[i] = dl;
        } else {
            throw std::invalid_argument(
                "Shapes " + lhs.to_string() + " and " +
                rhs.to_string() + " are not broadcastable.");
        }
    }

    return result;
}


Tensor Tensor::cat(const std::vector<Tensor>& tensors, int dim) {
    if (tensors.empty())
        throw std::invalid_argument("cat(): expected a non-empty list of tensors");

    auto& first = tensors.front();
    first.throw_if_not_valid();

    auto rank = first.ndim();
    dim = normalize_dim("cat()", dim, rank);
    
    auto ctx = first.ctx_;
    auto type = first.dtype();

    for (auto& tensor : tensors) {
        tensor.throw_if_not_valid();

        if (tensor.dtype() != type)
            throw std::invalid_argument("cat(): all tensors must have the same dtype");

        if (tensor.ndim() != rank)
            throw std::invalid_argument("cat(): tensors must have the same number of dimensions");

        for (int d = 0; d < rank; ++d) {
            if (d != dim && tensor.shape()[d] != first.shape()[d]) {
                throw std::invalid_argument(
                    "cat(): tensor sizes must match except in the concatenation dimension");
            }
        }
    }

    // Perform concatenation
    auto tensor = *first;
    for (auto i = 1; i < tensors.size(); ++i)
        tensor = ggml_concat(ctx, tensor, tensors[i].t_, rank - 1 - dim);

    // Calculate the correct output shape using PyTorch indexing
    Shape shape(rank);
    for (int i = 0; i < rank; ++i) {
        if (i == dim) {
            int64_t total_size = 0;
            for (const auto& tensor : tensors)
                total_size += tensor.shape_[i];
            shape[i] = total_size;
        } else
            shape[i] = first.shape_[i];
    }

    return Tensor(ctx, tensor, shape);
}

Tensor Tensor::stack(const std::vector<Tensor>& tensors, int64_t dim) {
    if (tensors.empty())
        throw std::invalid_argument("stack(): expected at least one tensor");

    const int64_t rank = tensors[0].ndim();

    // stack adds a new dimension, so valid insertion positions are
    // [-rank-1, rank].
    dim = normalize_dim("stack()", dim, rank, true);

    std::vector<Tensor> expanded;
    expanded.reserve(tensors.size());

    for (const auto& tensor : tensors) {
        if (tensor.shape() != tensors[0].shape())
            throw std::invalid_argument(
                "stack(): all tensors must have the same shape");

        expanded.push_back(tensor.unsqueeze(dim));
    }

    return Tensor::cat(expanded, dim);
}

Tensor Tensor::reshape(const Shape& shape) const {
    throw_if_not_valid();

    Shape out(shape);

    int infer_dim = -1;
    int64_t known_product = 1;

    for (size_t i = 0; i < out.rank(); ++i) {
        const int64_t dim = out[i];

        if (dim == -1) {
            if (infer_dim != -1)
                throw std::invalid_argument(
                    "reshape(): only one dimension may be inferred");

            infer_dim = static_cast<int>(i);
        } else {
            if (dim < 0)
                throw std::invalid_argument(
                    "reshape(): dimensions must be >= 0 or -1, not " + std::to_string(dim));

            if (dim != 0 &&
                known_product > std::numeric_limits<int64_t>::max() / dim)
                throw std::overflow_error("reshape(): shape is too large");

            known_product *= dim;
        }
    }

    const int64_t numel = this->numel();

    if (infer_dim != -1) {
        if (known_product == 0 || numel % known_product != 0)
            throw std::invalid_argument(
                "reshape(): shape is incompatible with tensor size");

        out[infer_dim] = numel / known_product;
    } else if (known_product != numel) {
        throw std::invalid_argument(
            "reshape(): shape is incompatible with tensor size");
    }

    auto src = *this;

    // ggml_reshape expects the tensor to be contiguous.
    if (!src.is_contiguous())
        src = src.contiguous();

    switch (out.rank()) {
    case 0:
        // GGML scalar == 1D tensor with one element.
        return Tensor(ctx_, 
            ggml_reshape_1d(ctx_, *src, 1),
            out);

    case 1:
        return Tensor(ctx_,
            ggml_reshape_1d(ctx_, *src, out.ne_[0]),
            out);

    case 2:
        return Tensor(ctx_,
            ggml_reshape_2d(ctx_, *src, out.ne_[0], out.ne_[1]),
            out);

    case 3:
        return Tensor(ctx_,
            ggml_reshape_3d(ctx_, *src, out.ne_[0], out.ne_[1], out.ne_[2]),
            out);

    case 4:
        return Tensor(ctx_,
            ggml_reshape_4d(ctx_, *src, out.ne_[0], out.ne_[1], out.ne_[2], out.ne_[3]),
            out);
    }

    throw std::invalid_argument("Unsupported shape: " + shape.to_string());
}

Tensor Tensor::permute(const Shape& order) const {
    throw_if_not_valid();

    auto rank = ndim();

    if (order.rank() != static_cast<size_t>(rank)) {
        throw std::invalid_argument(
            "permute(): order must specify exactly one dimension for every tensor dimension");
    }

    // Build output shape in logical (PyTorch) order
    Shape out(rank);
    std::array<bool, 4> seen = {false, false, false, false};
    std::vector<int> logical_axes(rank);

    for (int i = 0; i < rank; ++i) {
        auto axis = order[i];

        if (axis < 0)
            axis += rank;

        if (axis < 0 || axis >= rank || seen[axis])
            throw std::invalid_argument(
                "permute(): order must be a permutation of tensor dimensions");

        seen[axis] = true;
        logical_axes[i] = axis;
        out[i] = shape_[axis];
    }
    
    // Build inverse permutation for GGML
    // ggml_permute(p0,p1,p2,p3) means: old axis i goes to new position p_i
    // We have: new_logical_pos j gets data from old_logical_axis = logical_axes[j]
    // So: old_logical_axis -> new_logical_pos mapping is: logical_axes[j] -> j
    // Convert to GGML indices:
    std::array<int, 4> ggml_permute_args = {0, 1, 2, 3};
    for (int new_logical_pos = 0; new_logical_pos < rank; ++new_logical_pos) {
        int old_logical_axis = logical_axes[new_logical_pos];
        int old_ggml_axis = rank - 1 - old_logical_axis;
        int new_ggml_pos = rank - 1 - new_logical_pos;
        ggml_permute_args[old_ggml_axis] = new_ggml_pos;
    }

    return Tensor(ctx_, ggml_permute(ctx_, t_, 
                  ggml_permute_args[0], ggml_permute_args[1], 
                  ggml_permute_args[2], ggml_permute_args[3]), out);
}

Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
    throw_if_not_valid();

    auto rank = ndim();

    dim0 = normalize_dim("transpose", dim0, rank);
    dim1 = normalize_dim("transpose", dim1, rank);

    if (dim0 == dim1)
        return *this;

    Shape order(rank);

    for (int i = 0; i < rank; ++i)
        order[i] = i;

    std::swap(order[dim0], order[dim1]);

    return permute(order);
}

Tensor Tensor::squeeze(int64_t dim) const {
    throw_if_not_valid();

    auto rank = ndim();
    dim = normalize_dim("squeeze()", dim, rank);

    if (shape_[dim] != 1)
        throw std::invalid_argument("squeeze(): selected dimension must have size 1");

    // PyTorch: squeezing the only dimension of a (1,) tensor produces a scalar.
    Shape out(rank > 1 ? rank - 1 : 0);

    for (int src = 0, dst = 0; src < rank; ++src) {
        if (src == dim)
            continue;

        out[dst++] = shape_[src];
    }

    return reshape(out);
}

Tensor Tensor::unsqueeze(int64_t dim) const {
    throw_if_not_valid();

    auto rank = ndim();
    dim = normalize_dim("unsqueeze()", dim, rank, true);

    Shape out(rank + 1);

    for (auto src = 0, dst = 0; dst < rank + 1; ++dst) {
        if (dst == dim) {
            out[dst] = 1;
        } else {
            out[dst] = shape_[src++];
        }
    }

    return reshape(out);
}

Tensor Tensor::flatten(int64_t start_dim, int64_t end_dim) const {
    throw_if_not_valid();

    auto rank = ndim();

    start_dim = normalize_dim("flatten()", start_dim, rank);
    end_dim   = normalize_dim("flatten()", end_dim, rank);

    if (start_dim > end_dim)
        throw std::invalid_argument("flatten(): start_dim must be <= end_dim");

    Shape out(rank - (end_dim - start_dim));

    int dst = 0;

    for (int d = 0; d < start_dim; ++d)
        out[dst++] = shape_[d];

    int64_t flattened = 1;
    for (int d = start_dim; d <= end_dim; ++d) {
        if (shape_[d] != 0 &&
            flattened > std::numeric_limits<int64_t>::max() / shape_[d])
            throw std::overflow_error("flatten(): flattened dimension is too large");

        flattened *= shape_[d];
    }

    out[dst++] = flattened;

    for (int d = end_dim + 1; d < rank; ++d)
        out[dst++] = shape_[d];

    return reshape(out);
}

Tensor Tensor::unflatten(int64_t dim, const Shape& new_shape) {
    throw_if_not_valid();

    auto rank = ndim();
    auto target_dim = normalize_dim("unflatten()", dim, rank);

    auto out_rank = rank - 1 + new_shape.rank();

    if (out_rank > 4)
        throw std::invalid_argument("unflatten(): only at most 4 dimensions are supported, not " + std::to_string(out_rank));

    Shape out(out_rank);

    int64_t dst = 0;

    for (int d = 0; d < target_dim; ++d)
        out[dst++] = shape_[d];

    int64_t infer_dim = -1;
    int64_t known_product = 1;

    for (size_t i = 0; i < new_shape.rank(); ++i) {
        const int64_t value = new_shape[i];

        if (value == -1) {
            if (infer_dim != -1)
                throw std::invalid_argument(
                    "unflatten(): only one replacement dimension may be inferred");

            infer_dim = dst;
            out[dst++] = 1;
        } else {
            if (value < 0)
                throw std::invalid_argument(
                    "unflatten(): dimensions must be >= 0 or -1");

            if (value != 0 &&
                known_product > std::numeric_limits<int64_t>::max() / value)
                throw std::overflow_error("unflatten(): shape is too large");

            known_product *= value;
            out[dst++] = value;
        }
    }

    const int64_t old_size = shape_[target_dim];

    if (infer_dim != -1) {
        if (known_product == 0 || old_size % known_product != 0)
            throw std::invalid_argument(
                "unflatten(): replacement shape is incompatible with selected dimension");

        out[infer_dim] = old_size / known_product;
    } else if (known_product != old_size) {
        throw std::invalid_argument(
            "unflatten(): replacement shape is incompatible with selected dimension");
    }

    for (int64_t d = target_dim + 1; d < rank; ++d)
        out[dst++] = shape_[d];

    return reshape(out);
}

Tensor Tensor::narrow(int64_t dim, int64_t start, int64_t length) const {
    throw_if_not_valid();

    auto rank = ndim();
    dim = normalize_dim("narrow()", dim, rank);

    const int64_t dim_size = shape_[dim];

    if (start < 0)
        start += dim_size;

    if (start < 0 || start > dim_size)
        throw std::out_of_range("narrow(): start is out of range");

    if (length < 0 || length > dim_size - start)
        throw std::out_of_range("narrow(): length is out of range");

    Shape out = shape_;
    out[dim] = length;

    const size_t offset = static_cast<size_t>(start) * t_->nb[rank - 1 - dim];

    switch (rank) {
    case 1:
        return Tensor(ctx_, ggml_view_1d(ctx_, t_, out.ne_[0], offset), out);

    case 2:
        return Tensor(ctx_, ggml_view_2d(ctx_, t_, out.ne_[0], out.ne_[1], t_->nb[1], offset), out);

    case 3:
        return Tensor(ctx_, ggml_view_3d(ctx_, t_, out.ne_[0], out.ne_[1], out.ne_[2], t_->nb[1], t_->nb[2], offset), out);

    case 4:
        return Tensor(ctx_, ggml_view_4d(ctx_, t_, out.ne_[0], out.ne_[1], out.ne_[2], out.ne_[3], t_->nb[1], t_->nb[2], t_->nb[3], offset), out);
    }

    throw std::runtime_error("narrow(): invalid rank");
}

std::vector<Tensor> Tensor::chunk(int n, int64_t dim) const {
    throw_if_not_valid();

    if (n <= 0)
        throw std::invalid_argument("chunk(): n must be greater than zero");

    dim = normalize_dim("chunk()", dim, ndim());

    const int64_t dim_size = shape_[dim];

    if (dim_size == 0)
        return { narrow(dim, 0, 0) };

    // PyTorch's chunk chooses ceil(dim_size / n), and may return fewer than n chunks.
    const int64_t split_size = (dim_size + n - 1) / n;

    return split(split_size, dim);
}

std::vector<Tensor> Tensor::split(int64_t split_size, int64_t dim) const {
    throw_if_not_valid();

    if (split_size <= 0)
        throw std::invalid_argument("split(): split_size must be greater than zero");

    dim = normalize_dim("split()", dim, ndim());

    const int64_t dim_size = shape_[dim];

    if (dim_size == 0)
        return { narrow(dim, 0, 0) };

    std::vector<Tensor> result;
    result.reserve((dim_size + split_size - 1) / split_size);

    for (int64_t start = 0; start < dim_size; start += split_size)
        result.push_back(narrow(dim, start, std::min(split_size, dim_size - start)));

    return result;
}

std::vector<Tensor> Tensor::split_with_sizes(const std::vector<int64_t>& split_sizes, int64_t dim) const {
    throw_if_not_valid();

    if (split_sizes.empty())
        throw std::invalid_argument("split_with_sizes(): split_sizes must not be empty");

    dim = normalize_dim("split_with_sizes()", dim, ndim());
    const int64_t dim_size = shape_[dim];

    // Validate and sum sizes
    int64_t total = 0;
    for (const int64_t size : split_sizes) {
        if (size < 0)
            throw std::invalid_argument("split_with_sizes(): split sizes must be non-negative");
        
        total += size;
        if (total < 0)
            throw std::overflow_error("split_with_sizes(): sum of split sizes overflowed");
    }

    if (total != dim_size)
        throw std::invalid_argument("split_with_sizes(): split sizes must sum exactly to the size of the selected dimension");

    // Build result
    std::vector<Tensor> result;
    result.reserve(split_sizes.size());

    int64_t start = 0;
    for (const int64_t size : split_sizes) {
        result.push_back(narrow(dim, start, size));
        start += size;
    }

    return result;
}

Tensor Tensor::expand(const Shape& new_shape) const {
    throw_if_not_valid();

    // Expanding to a scalar is a no-op.
    if (new_shape.rank() == 0) {
        if (shape().rank() != 0) {
            throw std::runtime_error(
                "expand(): cannot expand a non-scalar to a scalar");
        }

        return *this;
    }

    auto src = *this;

    // Pad rank with leading PyTorch singleton dimensions.
    while (src.shape().rank() < new_shape.rank())
        src = src.unsqueeze(0);

    if (src.shape().rank() != new_shape.rank())
        throw std::runtime_error(
            "expand(): target rank must not be smaller than tensor rank");

    Shape repeats(new_shape.rank());

    for (int64_t i = 0; i < new_shape.rank(); ++i) {
        const int64_t current = src.shape()[i];
        const int64_t target = new_shape[i];

        if (current == target) {
            repeats[i] = 1;
        } else if (current == 1) {
            repeats[i] = target;
        } else {
            throw std::runtime_error(
                "expand(): incompatible dimension");
        }
    }

    return src.repeat(repeats);
}

#if 0
// without ggml_repeat

Tensor Tensor::repeat(const Shape& repeats) const {
    throw_if_not_valid();

    const int64_t rank = ndim();

    if (repeats.rank() != rank)
        throw std::invalid_argument(
            "repeat(): number of repeat dimensions must match tensor rank");

    auto result = *this;

    for (int64_t dim = 0; dim < rank; ++dim) {
        const int64_t count = repeats[dim];

        if (count < 0)
            throw std::invalid_argument(
                "repeat(): repeat dimensions must be non-negative");

        if (count == 0) {
            Shape out = result.shape();
            out[dim] = 0;

            return Tensor::empty(ctx_, out, result.dtype());
        }

        if (count == 1)
            continue;

        const auto original = result;
        const int64_t original_size = result.shape()[dim];

        // Build the required repetitions using powers of two.
        // This requires O(log(count)) concat operations.
        std::vector<Tensor> parts;

        int64_t remaining = count;
        int64_t power = 1;
        Tensor block = original;

        while (remaining > 0) {
            if (remaining & 1)
                parts.push_back(block);

            remaining >>= 1;

            if (remaining == 0)
                break;

            block = Tensor::cat({block, block}, dim);
            power <<= 1;
        }

        result = Tensor::cat(parts, dim);
    }

    return result;
}

#else

Tensor Tensor::repeat(const Shape& repeats) const {
    throw_if_not_valid();

    const int64_t rank = ndim();

    if (repeats.rank() != rank)
        throw std::invalid_argument(
            "repeat(): number of repeat dimensions must match tensor rank");

    Shape out(rank);

    for (int64_t i = 0; i < rank; ++i) {
        if (repeats[i] < 0)
            throw std::invalid_argument(
                "repeat(): repeat dimensions must be non-negative");

        if (shape_[i] == 0)
            out[i] = 0;
        else
            out[i] = shape_[i] * repeats[i];
    }

    // GGML repeat requires the target tensor to describe the
    // desired output shape.
    auto target = empty(ctx_, out, dtype());

    return Tensor(
        ctx_,
        ggml_repeat(ctx_, t_, *target),
        out
    );
}

#endif

Tensor Tensor::sum(int64_t dim, bool keepdim) const {
    throw_if_not_valid();

    const int64_t rank = ndim();
    dim = normalize_dim("sum()", dim, rank, false);

    // Move reduction dimension to the last axis.
    Shape order(rank);

    int64_t dst = 0;

    for (auto src = 0; src < rank; ++src) {
        if (src != dim)
            order[dst++] = src;
    }

    order[dst] = dim;

    auto x = *this;

    if (dim != rank - 1)
        x = permute(order);

    Shape out(rank - 1);

    for (int64_t src = 0, dst = 0; src < rank; ++src) {
        if (src != dim)
            out[dst++] = shape_[src];
    }

    // Tensor is required to be contiguous to sum the correct rows.
    if (!x.is_contiguous())
        x = x.contiguous();

    // GGML sum_rows reduces ne0 (fastest dimension).
    auto y = Tensor(x.ctx_, ggml_sum_rows(x.ctx_, *x), out);

    if (keepdim)
        y = y.unsqueeze(dim);

    return y;
}

Tensor Tensor::mean(int64_t dim, bool keepdim) const {
    const int64_t size = shape_[dim];
    auto t = sum(dim, keepdim);

    return t / static_cast<float>(size);
}

Tensor Tensor::operator[](const std::vector<Slice>& indices) const {
    throw_if_not_valid();

    auto input_rank = ndim();

    int ellipsis_count = 0;
    int explicitly_consumed_dims = 0;

    for (const Slice& item : indices) {
        switch (item.kind) {
            case Slice::Kind::All:
            case Slice::Kind::Index:
            case Slice::Kind::Range:
                ++explicitly_consumed_dims;
                break;

            case Slice::Kind::Ellipsis:
                ++ellipsis_count;
                break;

            default:
                break;
        }
    }

    if (ellipsis_count > 1)
        throw std::invalid_argument(
            "operator[]: only one ellipsis is allowed");

    if (explicitly_consumed_dims > input_rank)
        throw std::invalid_argument(
            "operator[]: too many indices for tensor");

    auto implicit_all_dims = input_rank - explicitly_consumed_dims;

    std::vector<Slice> expanded;
    expanded.reserve(indices.size() + implicit_all_dims);

    bool has_ellipsis = false;

    for (const Slice& item : indices) {
        if (item.kind == Slice::Kind::Ellipsis) {
            for (int i = 0; i < implicit_all_dims; ++i)
                expanded.push_back(Slice::all());

            has_ellipsis = true;
        } else
            expanded.push_back(item);
    }

    // Python/PyTorch implicitly append ':' for unmentioned input dimensions.
    if (!has_ellipsis) {
        for (int i = 0; i < implicit_all_dims; ++i)
            expanded.push_back(Slice::all());
    }

    auto result = *this;
    auto output_dim = 0;

    for (const Slice& item : expanded) {
        switch (item.kind) {
            case Slice::Kind::NewAxis:
                result = result.unsqueeze(output_dim);
                ++output_dim;
                break;

            case Slice::Kind::All:
                ++output_dim;
                break;

            case Slice::Kind::Index: {
                const int64_t dim_size = result.shape()[output_dim];

                int64_t normalized_index = item.start.value_or(0);
                if (normalized_index < 0)
                    normalized_index += dim_size;

                if (normalized_index < 0 || normalized_index >= dim_size)
                    throw std::out_of_range("operator[]: index out of range");

                result = result.narrow(output_dim, normalized_index, 1);
                result = result.squeeze(output_dim);
                // Do not increment output_dim: integer indexing removed it.
                break;
            }

            case Slice::Kind::Range: {
                if (item.step != 1) {
                    throw std::invalid_argument(
                        "operator[]: slice steps other than 1 are not supported");
                }

                const int64_t dim_size = result.shape()[output_dim];

                int64_t normalized_start = item.start.value_or(0);
                int64_t normalized_stop  = item.stop.value_or(dim_size);

                if (normalized_start < 0)
                    normalized_start += dim_size;

                if (normalized_stop < 0)
                    normalized_stop += dim_size;

                normalized_start = std::clamp<int64_t>(normalized_start, 0, dim_size);
                normalized_stop  = std::clamp<int64_t>(normalized_stop, 0, dim_size);

                // Example: x[5:2] becomes an empty slice.
                if (normalized_stop < normalized_start)
                    normalized_stop = normalized_start;

                result = result.narrow(
                    output_dim,
                    normalized_start,
                    normalized_stop - normalized_start);

                ++output_dim;
                break;
            }

            default:
                break;
        }
    }

    return result;
}

int Tensor::normalize_dim(const std::string& method, int64_t dim, int64_t rank, bool allow_end) {
    const int64_t upper = allow_end ? rank : rank - 1;
    const int64_t lower = allow_end ? -(rank + 1) : -rank;

    if (dim < lower || dim > upper)
        throw std::out_of_range(method + ": dimension out of range");

    if (dim < 0)
        dim += rank + (allow_end ? 1 : 0);

    return dim;
}

ggml_type Tensor::common_dtype(ggml_type a, ggml_type b) {
    // Quantized types cannot generally participate directly in
    // elementwise binary arithmetic. Promote them to a floating-point
    // computation type.
    //
    // Q + Q       -> F32
    // Q + F64     -> F64
    // Q + F32     -> F32
    // Q + F16     -> F16
    // Q + BF16    -> BF16
    // Q + integer -> F32
    const bool a_quantized = ggml_is_quantized(a);
    const bool b_quantized = ggml_is_quantized(b);

    if (a_quantized || b_quantized) {
        if (a_quantized && b_quantized)
            return GGML_TYPE_F32;

        const auto other = a_quantized ? b : a;

        if (other == GGML_TYPE_F64)
            return GGML_TYPE_F64;

        if (other == GGML_TYPE_F32)
            return GGML_TYPE_F32;

        if (other == GGML_TYPE_F16)
            return GGML_TYPE_F16;

        if (other == GGML_TYPE_BF16)
            return GGML_TYPE_BF16;

        // Quantized + integer has no useful common integer
        // representation, so compute in F32.
        return GGML_TYPE_F32;
    }

    // Same type
     if (a == b)
        return a;

    // Floating point
    if (a == GGML_TYPE_F64 || b == GGML_TYPE_F64)
        return GGML_TYPE_F64;

    if (a == GGML_TYPE_F32 || b == GGML_TYPE_F32)
        return GGML_TYPE_F32;

    // float16 + bfloat16 to float32
    if ((a == GGML_TYPE_F16  && b == GGML_TYPE_BF16) ||
        (a == GGML_TYPE_BF16 && b == GGML_TYPE_F16))
        return GGML_TYPE_F32;

    if (a == GGML_TYPE_F16 || b == GGML_TYPE_F16)
        return GGML_TYPE_F16;

    if (a == GGML_TYPE_BF16 || b == GGML_TYPE_BF16)
        return GGML_TYPE_BF16;

    // Signed integers: wider type wins
    if (a == GGML_TYPE_I64 || b == GGML_TYPE_I64)
        return GGML_TYPE_I64;

    if (a == GGML_TYPE_I32 || b == GGML_TYPE_I32)
        return GGML_TYPE_I32;

    if (a == GGML_TYPE_I16 || b == GGML_TYPE_I16)
        return GGML_TYPE_I16;

    if (a == GGML_TYPE_I8 || b == GGML_TYPE_I8)
        return GGML_TYPE_I8;

    throw std::runtime_error("Unsupported GGML types: " + std::string(ggml_type_name(a)) + " and " + std::string(ggml_type_name(b)));
}

void Tensor::throw_if_not_valid() const {
    if (!ctx_ || !t_)
        throw std::runtime_error("undefined Tensor");
}
