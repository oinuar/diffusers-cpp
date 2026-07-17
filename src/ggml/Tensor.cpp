#include "ggml/Tensor.hpp"
#include <limits>
#include <sstream>
#include <algorithm>
#include <iostream>

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

    auto src = clone_as_contiguous();

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
    
    auto src = clone_as_contiguous();

    return Tensor(ctx_, ggml_permute(ctx_, *src, 
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
        throw std::invalid_argument("unflatten(): only at most 4 dimensions are supported");

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

    auto src = clone_as_contiguous();
    const size_t offset = static_cast<size_t>(start) * src.t_->nb[rank - 1 - dim];

    switch (rank) {
    case 1:
        return Tensor(ctx_, ggml_view_1d(ctx_, *src, out.ne_[0], offset), out);

    case 2:
        return Tensor(ctx_, ggml_view_2d(ctx_, *src, out.ne_[0], out.ne_[1], src.t_->nb[1], offset), out);

    case 3:
        return Tensor(ctx_, ggml_view_3d(ctx_, *src, out.ne_[0], out.ne_[1], out.ne_[2], src.t_->nb[1], src.t_->nb[2], offset), out);

    case 4:
        return Tensor(ctx_, ggml_view_4d(ctx_, *src, out.ne_[0], out.ne_[1], out.ne_[2], out.ne_[3], src.t_->nb[1], src.t_->nb[2], src.t_->nb[3], offset), out);
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

    // Pad rank with leading PyTorch singleton dimensions
    while (src.shape().rank() < new_shape.rank())
        src = src.unsqueeze(0);

    // Validate
    for (int i = 0; i < new_shape.rank(); ++i) {
        auto cur = src.shape()[i];
        auto dst = new_shape[i];

        if (cur != dst && cur != 1) {
            throw std::runtime_error(
                "expand(): incompatible dimension");
        }
    }

    // Create a dummy tensor with the target shape to use as the repeat target.
    // GGML's ggml_repeat only reads the 'ne' array of the target tensor, 
    // so the type (GGML_TYPE_F32) and lack of a data buffer are perfectly safe.
    auto target =
        ggml_new_tensor(ctx_, src.t_->type,
                        new_shape.rank(),
                        new_shape.data());

    return Tensor(
        ctx_,
        ggml_repeat(ctx_, src.t_, target),
        new_shape);
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
    auto cloned = false;

    for (const Slice& item : expanded) {
        switch (item.kind) {
            case Slice::Kind::NewAxis:
                result = result.unsqueeze(output_dim);
                cloned = true;
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
                cloned = true;
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

                cloned = true;
                ++output_dim;
                break;
            }

            default:
                break;
        }
    }

    // Clone tensor if not already cloned
    if (!cloned)
        result = result.clone();

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

void Tensor::throw_if_not_valid() const {
    if (!ctx_ || !t_)
        throw std::runtime_error("undefined Tensor");
}
