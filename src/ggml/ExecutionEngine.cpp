#include "ggml/ExecutionEngine.hpp"
#include "ggml/Scope.hpp"
#include "ggml/Context.hpp"

ExecutionEngine ExecutionEngine::Default;

// -------------------------------------------------------------------------
// Tensor creation / initialization
// -------------------------------------------------------------------------

ggml_tensor* ExecutionEngine::new_tensor(
    ggml_context* ctx,
    ggml_type type,
    int n_dims,
    const int64_t* ne
) {
    return ggml_new_tensor(ctx, type, n_dims, ne);
}

ggml_tensor* ExecutionEngine::new_tensor_1d(
    ggml_context* ctx,
    ggml_type type,
    int64_t ne0
) {
    return ggml_new_tensor_1d(ctx, type, ne0);
}

void ExecutionEngine::set_input(
    ggml_tensor* tensor
) {
    ggml_set_input(tensor);
}

ggml_tensor* ExecutionEngine::fill(
    ggml_tensor* tensor,
    float value
) {
    return ggml_fill(*Scope::context(), tensor, value);
}

// -----------------------------------------------------------------------------
// Copy / cast
// -----------------------------------------------------------------------------

ggml_tensor* ExecutionEngine::cont(
    ggml_tensor* tensor
) {
    return ggml_cont(*Scope::context(), tensor);
}

ggml_tensor* ExecutionEngine::dup(
    ggml_tensor* tensor
) {
    return ggml_dup(*Scope::context(), tensor);
}

ggml_tensor* ExecutionEngine::cast(
    ggml_tensor* tensor,
    ggml_type type
) {
    return ggml_cast(*Scope::context(), tensor, type);
}

ggml_tensor* ExecutionEngine::cpy(
    ggml_tensor* src,
    ggml_tensor* dst
) {
    return ggml_cpy(*Scope::context(), src, dst);
}

// -----------------------------------------------------------------------------
// Unary arithmetic
// -----------------------------------------------------------------------------

ggml_tensor* ExecutionEngine::neg(
    ggml_tensor* tensor
) {
    return ggml_neg(*Scope::context(), tensor);
}

ggml_tensor* ExecutionEngine::abs(
    ggml_tensor* tensor
) {
    return ggml_abs(*Scope::context(), tensor);
}

ggml_tensor* ExecutionEngine::sqrt(
    ggml_tensor* tensor
) {
    return ggml_sqrt(*Scope::context(), tensor);
}

ggml_tensor* ExecutionEngine::exp(
    ggml_tensor* tensor
) {
    return ggml_exp(*Scope::context(), tensor);
}

ggml_tensor* ExecutionEngine::log(
    ggml_tensor* tensor
) {
    return ggml_log(*Scope::context(), tensor);
}

ggml_tensor* ExecutionEngine::sin(
    ggml_tensor* tensor
) {
    return ggml_sin(*Scope::context(), tensor);
}

ggml_tensor* ExecutionEngine::cos(
    ggml_tensor* tensor
) {
    return ggml_cos(*Scope::context(), tensor);
}

// -----------------------------------------------------------------------------
// Binary arithmetic
// -----------------------------------------------------------------------------

ggml_tensor* ExecutionEngine::add(
    ggml_tensor* lhs,
    ggml_tensor* rhs
) {
    return ggml_add(*Scope::context(), lhs, rhs);
}

ggml_tensor* ExecutionEngine::sub(
    ggml_tensor* lhs,
    ggml_tensor* rhs
) {
    return ggml_sub(*Scope::context(), lhs, rhs);
}

ggml_tensor* ExecutionEngine::mul(
    ggml_tensor* lhs,
    ggml_tensor* rhs
) {
    return ggml_mul(*Scope::context(), lhs, rhs);
}

ggml_tensor* ExecutionEngine::div(
    ggml_tensor* lhs,
    ggml_tensor* rhs
) {
    return ggml_div(*Scope::context(), lhs, rhs);
}

// -----------------------------------------------------------------------------
// Scalar operations
// -----------------------------------------------------------------------------

ggml_tensor* ExecutionEngine::scale(
    ggml_tensor* tensor,
    float value
) {
    return ggml_scale(*Scope::context(), tensor, value);
}

ggml_tensor* ExecutionEngine::clamp(
    ggml_tensor* tensor,
    float min,
    float max
) {
    return ggml_clamp(*Scope::context(), tensor, min, max);
}

// -----------------------------------------------------------------------------
// Matrix operations
// -----------------------------------------------------------------------------

ggml_tensor* ExecutionEngine::mul_mat(
    ggml_tensor* lhs,
    ggml_tensor* rhs
) {
    return ggml_mul_mat(*Scope::context(), lhs, rhs);
}

// -----------------------------------------------------------------------------
// Reshape
// -----------------------------------------------------------------------------

ggml_tensor* ExecutionEngine::reshape_1d(
    ggml_tensor* tensor,
    int64_t ne0
) {
    return ggml_reshape_1d(*Scope::context(), tensor, ne0);
}

ggml_tensor* ExecutionEngine::reshape_2d(
    ggml_tensor* tensor,
    int64_t ne0,
    int64_t ne1
) {
    return ggml_reshape_2d(*Scope::context(), tensor, ne0, ne1);
}

ggml_tensor* ExecutionEngine::reshape_3d(
    ggml_tensor* tensor,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2
) {
    return ggml_reshape_3d(*Scope::context(), tensor, ne0, ne1, ne2);
}

ggml_tensor* ExecutionEngine::reshape_4d(
    ggml_tensor* tensor,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2,
    int64_t ne3
) {
    return ggml_reshape_4d(
        *Scope::context(),
        tensor,
        ne0,
        ne1,
        ne2,
        ne3
    );
}

// -----------------------------------------------------------------------------
// Permute / transpose
// -----------------------------------------------------------------------------

ggml_tensor* ExecutionEngine::permute(
    ggml_tensor* tensor,
    int axis0,
    int axis1,
    int axis2,
    int axis3
) {
    return ggml_permute(
        *Scope::context(),
        tensor,
        axis0,
        axis1,
        axis2,
        axis3
    );
}

// -----------------------------------------------------------------------------
// Views
// -----------------------------------------------------------------------------

ggml_tensor* ExecutionEngine::view_1d(
    ggml_tensor* tensor,
    int64_t ne0,
    size_t offset
) {
    return ggml_view_1d(
        *Scope::context(),
        tensor,
        ne0,
        offset
    );
}

ggml_tensor* ExecutionEngine::view_2d(
    ggml_tensor* tensor,
    int64_t ne0,
    int64_t ne1,
    size_t nb1,
    size_t offset
) {
    return ggml_view_2d(
        *Scope::context(),
        tensor,
        ne0,
        ne1,
        nb1,
        offset
    );
}

ggml_tensor* ExecutionEngine::view_3d(
    ggml_tensor* tensor,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2,
    size_t nb1,
    size_t nb2,
    size_t offset
) {
    return ggml_view_3d(
        *Scope::context(),
        tensor,
        ne0,
        ne1,
        ne2,
        nb1,
        nb2,
        offset
    );
}

ggml_tensor* ExecutionEngine::view_4d(
    ggml_tensor* tensor,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2,
    int64_t ne3,
    size_t nb1,
    size_t nb2,
    size_t nb3,
    size_t offset
) {
    return ggml_view_4d(
        *Scope::context(),
        tensor,
        ne0,
        ne1,
        ne2,
        ne3,
        nb1,
        nb2,
        nb3,
        offset
    );
}

// -----------------------------------------------------------------------------
// Repeat
// -----------------------------------------------------------------------------

ggml_tensor* ExecutionEngine::repeat(
    ggml_tensor* tensor,
    ggml_tensor* target
) {
    return ggml_repeat(*Scope::context(), tensor, target);
}

// -----------------------------------------------------------------------------
// Concatenation
// -----------------------------------------------------------------------------

ggml_tensor* ExecutionEngine::concat(
    ggml_tensor* a,
    ggml_tensor* b,
    int dim
) {
    return ggml_concat(*Scope::context(), a, b, dim);
}

// -----------------------------------------------------------------------------
// Reduction
// -----------------------------------------------------------------------------

ggml_tensor* ExecutionEngine::sum_rows(
    ggml_tensor* tensor
) {
    return ggml_sum_rows(*Scope::context(), tensor);
}

ggml_tensor* ExecutionEngine::silu(
    ggml_tensor* tensor
) {
    return ggml_silu(*Scope::context(), tensor);
}
