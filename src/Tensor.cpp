#include "Tensor.hpp"
#include <limits>
#include <sstream>

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

Tensor Tensor::cat(const std::vector<Tensor>& tensors, int dim) {
    if (tensors.empty()) {
        throw std::invalid_argument("cat(): expected a non-empty list of tensors");
    }

    const Tensor& first = tensors.front();
    first.throw_if_not_valid();

    const int rank = first.ndim();
    dim = normalize_dim(dim, rank);

    ggml_context* ctx = first.ctx_;
    ggml_type type = first.dtype();

    for (const Tensor& tensor : tensors) {
        tensor.throw_if_not_valid();

        if (tensor.ctx_ != ctx) {
            throw std::invalid_argument("cat(): all tensors must belong to the same ggml context");
        }

        if (tensor.dtype() != type) {
            throw std::invalid_argument("cat(): all tensors must have the same dtype");
        }

        if (tensor.ndim() != rank) {
            throw std::invalid_argument("cat(): tensors must have the same number of dimensions");
        }

        for (int d = 0; d < rank; ++d) {
            if (d != dim && tensor.t_->ne[d] != first.t_->ne[d]) {
                throw std::invalid_argument(
                    "cat(): tensor sizes must match except in the concatenation dimension");
            }
        }
    }

    Tensor result = first;
    for (size_t i = 1; i < tensors.size(); ++i) {
        result = Tensor(ctx, ggml_concat(ctx, result.t_, tensors[i].t_, dim));
    }

    return result;
}

Tensor Tensor::reshape(const Shape& shape) const {
    throw_if_not_valid();
    throw_if_not_contiguous();

    std::array<int64_t, 4> out = {1, 1, 1, 1};

    int infer_dim = -1;
    int64_t known_product = 1;

    for (size_t i = 0; i < shape.rank(); ++i) {
        const int64_t value = shape[i];

        if (value == -1) {
            if (infer_dim != -1) {
                throw std::invalid_argument("reshape(): only one dimension may be inferred");
            }

            infer_dim = static_cast<int>(i);
        } else {
            if (value < 0) {
                throw std::invalid_argument("reshape(): dimensions must be >= 0 or -1");
            }

            if (value != 0 &&
                known_product > std::numeric_limits<int64_t>::max() / value) {
                throw std::overflow_error("reshape(): shape is too large");
            }

            known_product *= value;
            out[i] = value;
        }
    }

    const int64_t total = numel();

    if (infer_dim != -1) {
        if (known_product == 0) {
            throw std::invalid_argument(
                "reshape(): cannot infer a dimension when known dimensions multiply to zero");
        }

        if (total % known_product != 0) {
            throw std::invalid_argument(
                "reshape(): requested shape is incompatible with tensor size");
        }

        out[infer_dim] = total / known_product;
    } else if (known_product != total)
        throw std::invalid_argument(
            "reshape(): requested shape is incompatible with tensor size");

    switch (shape.rank()) {
        case 1:
            return Tensor(ctx_, ggml_reshape_1d(ctx_, t_, out[0]));
        case 2:
            return Tensor(ctx_, ggml_reshape_2d(ctx_, t_, out[0], out[1]));
        case 3:
            return Tensor(ctx_, ggml_reshape_3d(ctx_, t_, out[0], out[1], out[2]));
        default:
            break;
    }

    return Tensor(ctx_, ggml_reshape_4d(ctx_, t_, out[0], out[1], out[2], out[3]));
}

Tensor Tensor::permute(const Shape& order) const {
    throw_if_not_valid();

    const int rank = ndim();

    if (order.rank() != static_cast<size_t>(rank)) {
        throw std::invalid_argument(
            "permute(): order must specify exactly one dimension for every tensor dimension");
    }

    std::array<int, 4> axes = {0, 1, 2, 3};
    std::array<bool, 4> seen = {false, false, false, false};

    for (int i = 0; i < rank; ++i) {
        int axis = static_cast<int>(order[i]);

        if (axis < 0) {
            axis += rank;
        }

        if (axis < 0 || axis >= rank || seen[axis]) {
            throw std::invalid_argument(
                "permute(): order must be a permutation of tensor dimensions");
        }

        axes[i] = axis;
        seen[axis] = true;
    }

    return Tensor(ctx_, ggml_permute(ctx_, t_, axes[0], axes[1], axes[2], axes[3]));
}

Tensor Tensor::squeeze(int dim) const {
    throw_if_not_valid();

    const int rank = ndim();
    dim = normalize_dim(dim, rank);

    if (t_->ne[dim] != 1)
        throw std::invalid_argument(
            "squeeze(): selected dimension must have size 1");

    // ggml does not support true rank-0 tensors.
    if (rank == 1)
        return *this;

    std::array<int64_t, 4> ne = {1, 1, 1, 1};
    std::array<size_t, 4> nb = {0, 0, 0, 0};

    int dst = 0;
    for (int src = 0; src < rank; ++src) {
        if (src == dim)
            continue;

        ne[dst] = t_->ne[src];
        nb[dst] = t_->nb[src];
        ++dst;
    }

    switch (rank - 1) {
        case 1:
            return Tensor(ctx_, ggml_view_1d(ctx_, t_, ne[0], 0));
        case 2:
            return Tensor(ctx_, ggml_view_2d(ctx_, t_, ne[0], ne[1], nb[1], 0));
        case 3:
            return Tensor(ctx_, ggml_view_3d(ctx_, t_, ne[0], ne[1], ne[2], nb[1], nb[2], 0));
        default:
            break;
    }

    throw std::logic_error("squeeze(): unsupported tensor rank");
}

Tensor Tensor::unsqueeze(int dim) const {
    throw_if_not_valid();

    const int rank = ndim();
    dim = normalize_dim(dim, rank, true);

    std::array<int64_t, 4> out = {1, 1, 1, 1};

    for (int src = 0, dst = 0; dst < rank + 1; ++dst) {
        if (dst == dim) {
            out[dst] = 1;
        } else {
            out[dst] = t_->ne[src++];
        }
    }

    Shape result_shape;
    result_shape.ne_ = out;
    result_shape.rank_ = rank + 1;

    return reshape(result_shape);
}

Tensor Tensor::flatten(int start_dim, int end_dim) const {
    throw_if_not_valid();
    throw_if_not_contiguous();

    const int rank = ndim();

    start_dim = normalize_dim(start_dim, rank);
    end_dim   = normalize_dim(end_dim, rank);

    if (start_dim > end_dim)
        throw std::invalid_argument("flatten(): start_dim must be <= end_dim");

    std::array<int64_t, 4> out = {1, 1, 1, 1};
    int out_rank = 0;

    for (int d = 0; d < start_dim; ++d) {
        out[out_rank++] = t_->ne[d];
    }

    int64_t flattened = 1;
    for (int d = start_dim; d <= end_dim; ++d) {
        if (t_->ne[d] != 0 &&
            flattened > std::numeric_limits<int64_t>::max() / t_->ne[d]) {
            throw std::overflow_error("flatten(): flattened dimension is too large");
        }

        flattened *= t_->ne[d];
    }

    out[out_rank++] = flattened;

    for (int d = end_dim + 1; d < rank; ++d) {
        out[out_rank++] = t_->ne[d];
    }

    Shape result_shape;
    result_shape.ne_ = out;
    result_shape.rank_ = out_rank;

    return reshape(result_shape);
}

Tensor Tensor::unflatten(int64_t dim, const Shape& shape) {
    throw_if_not_valid();
    throw_if_not_contiguous();

    const int rank = ndim();
    const int target_dim = normalize_dim(static_cast<int>(dim), rank);

    const int new_rank = rank - 1 + static_cast<int>(shape.rank());
    if (new_rank > 4)
        throw std::invalid_argument("unflatten(): only at most 4 dimensions are supported");

    std::array<int64_t, 4> out = {1, 1, 1, 1};
    int out_rank = 0;

    for (int d = 0; d < target_dim; ++d) {
        out[out_rank++] = t_->ne[d];
    }

    int infer_dim = -1;
    int64_t known_product = 1;

    for (size_t i = 0; i < shape.rank(); ++i) {
        const int64_t value = shape[i];

        if (value == -1) {
            if (infer_dim != -1)
                throw std::invalid_argument(
                    "unflatten(): only one replacement dimension may be inferred");

            infer_dim = out_rank;
            out[out_rank++] = 1;
        } else {
            if (value < 0)
                throw std::invalid_argument(
                    "unflatten(): dimensions must be >= 0 or -1");

            if (value != 0 &&
                known_product > std::numeric_limits<int64_t>::max() / value) {
                throw std::overflow_error("unflatten(): shape is too large");
            }

            known_product *= value;
            out[out_rank++] = value;
        }
    }

    const int64_t old_size = t_->ne[target_dim];

    if (infer_dim != -1) {
        if (known_product == 0 || old_size % known_product != 0)
            throw std::invalid_argument(
                "unflatten(): replacement shape is incompatible with selected dimension");

        out[infer_dim] = old_size / known_product;
    } else if (known_product != old_size)
        throw std::invalid_argument(
            "unflatten(): replacement shape is incompatible with selected dimension");

    for (int d = target_dim + 1; d < rank; ++d) {
        out[out_rank++] = t_->ne[d];
    }

    Shape result_shape;
    result_shape.ne_ = out;
    result_shape.rank_ = out_rank;

    return reshape(result_shape);
}

Tensor Tensor::narrow(int dim, int64_t start, int64_t length) const {
    throw_if_not_valid();

    const int rank = ndim();
    dim = normalize_dim(dim, rank);

    const int64_t dim_size = t_->ne[dim];

    if (start < 0)
        start += dim_size;

    if (start < 0 || start > dim_size)
        throw std::out_of_range("narrow(): start is out of range");

    if (length < 0 || length > dim_size - start)
        throw std::out_of_range("narrow(): length is out of range");

    std::array<int64_t, 4> ne = {
        t_->ne[0], t_->ne[1], t_->ne[2], t_->ne[3]
    };
    ne[dim] = length;

    const size_t offset = static_cast<size_t>(start) * t_->nb[dim];

    switch (rank) {
        case 1:
            return Tensor(ctx_, ggml_view_1d(ctx_, t_, ne[0], offset));
        case 2:
            return Tensor( ctx_, ggml_view_2d(ctx_, t_, ne[0], ne[1], t_->nb[1], offset));
        case 3:
            return Tensor(ctx_, ggml_view_3d(ctx_, t_, ne[0], ne[1], ne[2], t_->nb[1], t_->nb[2], offset));
        default:
            break;
    }

    return Tensor(ctx_, ggml_view_4d(ctx_, t_, ne[0], ne[1], ne[2], ne[3], t_->nb[1], t_->nb[2], t_->nb[3], offset));
}

std::vector<Tensor> Tensor::chunk(int n, int dim) const {
    throw_if_not_valid();

    if (n <= 0)
        throw std::invalid_argument("chunk(): chunks must be greater than zero");

    dim = normalize_dim(dim, ndim());

    const int64_t dim_size = t_->ne[dim];

    if (dim_size == 0)
        return { narrow(dim, 0, 0) };

    // PyTorch's chunk chooses ceil(dim_size / n), and may return fewer than n chunks.
    const int64_t chunk_size = (dim_size + n - 1) / n;

    std::vector<Tensor> result;
    result.reserve(static_cast<size_t>((dim_size + chunk_size - 1) / chunk_size));

    for (int64_t start = 0; start < dim_size; start += chunk_size)
        result.push_back(narrow(dim, start, std::min(chunk_size, dim_size - start)));

    return result;
}

std::vector<Tensor> Tensor::split(int64_t split_size, int dim) const {
    throw_if_not_valid();

    if (split_size <= 0)
        throw std::invalid_argument("split(): split_size must be greater than zero");

    dim = normalize_dim(dim, ndim());

    const int64_t dim_size = t_->ne[dim];

    if (dim_size == 0) {
        return { narrow(dim, 0, 0) };
    }

    std::vector<Tensor> result;
    result.reserve(static_cast<size_t>((dim_size + split_size - 1) / split_size));

    for (int64_t start = 0; start < dim_size; start += split_size)
        result.push_back(narrow(dim, start, std::min(split_size, dim_size - start)));

    return result;
}

std::vector<Tensor> Tensor::split_with_sizes(const std::vector<int64_t>& split_sizes, int dim) const {
    throw_if_not_valid();

    const int rank = ndim();
    dim = normalize_dim(dim, rank);

    if (split_sizes.empty())
        throw std::invalid_argument(
            "split_with_sizes(): split_sizes must not be empty");

    const int64_t dim_size = t_->ne[dim];

    int64_t total = 0;
    for (const int64_t size : split_sizes) {
        if (size < 0) {
            throw std::invalid_argument(
                "split_with_sizes(): split sizes must be non-negative");
        }

        if (size > std::numeric_limits<int64_t>::max() - total)
            throw std::overflow_error(
                "split_with_sizes(): sum of split sizes overflowed");

        total += size;
    }

    if (total != dim_size)
        throw std::invalid_argument(
            "split_with_sizes(): split sizes must sum exactly to the size "
            "of the selected dimension");

    std::vector<Tensor> result;
    result.reserve(split_sizes.size());

    int64_t start = 0;

    for (const int64_t size : split_sizes) {
        result.push_back(narrow(dim, start, size));
        start += size;
    }

    return result;
}

Tensor Tensor::expand(const Tensor::Shape& new_shape) const {
    auto dst = ggml_view_tensor(ctx_, t_);
    auto ne = shape();

    for (auto i = 0; i < ne.rank(); ++i) {
        dst->ne[i] = new_shape[i];

        if (ne[i] == 1 && new_shape[i] > 1)
            dst->nb[i] = 0;
        else
            dst->nb[i] = t_->nb[i];
    }

    return Tensor(ctx_, dst);
}

Tensor Tensor::operator[](const std::vector<Slice>& indices) const {
    throw_if_not_valid();

    const int input_rank = ndim();

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

    const int implicit_all_dims = input_rank - explicitly_consumed_dims;

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

    Tensor result = *this;
    int output_dim = 0;

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

int Tensor::normalize_dim(int dim, int rank, bool allow_end) {
    const int upper = allow_end ? rank : rank - 1;
    const int lower = allow_end ? -(rank + 1) : -rank;

    if (dim < lower || dim > upper)
        throw std::out_of_range("dimension out of range");

    if (dim < 0)
        dim += rank + (allow_end ? 1 : 0);

    return dim;
}

void Tensor::throw_if_not_valid() const {
    if (!ctx_ || !t_)
        throw std::runtime_error("undefined Tensor");
}

void Tensor::throw_if_not_contiguous() const {
    if (!ggml_is_contiguous(t_))
        throw std::runtime_error(
            "a contiguous tensor is required; call contiguous() first");
}
