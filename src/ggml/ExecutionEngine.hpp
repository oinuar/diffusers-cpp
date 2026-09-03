#pragma once

#include "ggml/Engine.hpp"

class ExecutionEngine : public Engine {
public:
    static ExecutionEngine Default;

    // -------------------------------------------------------------------------
    // Tensor creation / initialization
    // -------------------------------------------------------------------------

    ggml_tensor* new_tensor(
        ggml_context* ctx,
        ggml_type type,
        int n_dims,
        const int64_t* ne
    ) override;

    ggml_tensor* new_tensor_1d(
        ggml_context* ctx,
        ggml_type type,
        int64_t ne0
    ) override;

    void set_input(
        ggml_tensor* tensor
    ) override;

    ggml_tensor* fill(
        ggml_tensor* tensor,
        float value
    ) override;

    // Copy / cast
    ggml_tensor* cont(
        ggml_tensor* tensor
    ) override;

    ggml_tensor* dup(
        ggml_tensor* tensor
    ) override;

    ggml_tensor* cast(
        ggml_tensor* tensor,
        ggml_type type
    ) override;

    ggml_tensor* cpy(
        ggml_tensor* src,
        ggml_tensor* dst
    ) override;

    // Unary
    ggml_tensor* neg(
        ggml_tensor* tensor
    ) override;

    ggml_tensor* abs(
        ggml_tensor* tensor
    ) override;

    ggml_tensor* sqrt(
        ggml_tensor* tensor
    ) override;

    ggml_tensor* exp(
        ggml_tensor* tensor
    ) override;

    ggml_tensor* log(
        ggml_tensor* tensor
    ) override;

    ggml_tensor* sin(
        ggml_tensor* tensor
    ) override;

    ggml_tensor* cos(
        ggml_tensor* tensor
    ) override;

    // Binary
    ggml_tensor* add(
        ggml_tensor* lhs,
        ggml_tensor* rhs
    ) override;

    ggml_tensor* sub(
        ggml_tensor* lhs,
        ggml_tensor* rhs
    ) override;

    ggml_tensor* mul(
        ggml_tensor* lhs,
        ggml_tensor* rhs
    ) override;

    ggml_tensor* div(
        ggml_tensor* lhs,
        ggml_tensor* rhs
    ) override;

    // Scalar
    ggml_tensor* scale(
        ggml_tensor* tensor,
        float value
    ) override;

    ggml_tensor* clamp(
        ggml_tensor* tensor,
        float min,
        float max
    ) override;

    // Matrix
    ggml_tensor* mul_mat(
        ggml_tensor* lhs,
        ggml_tensor* rhs
    ) override;

    // Reshape
    ggml_tensor* reshape_1d(
        ggml_tensor* tensor,
        int64_t ne0
    ) override;

    ggml_tensor* reshape_2d(
        ggml_tensor* tensor,
        int64_t ne0,
        int64_t ne1
    ) override;

    ggml_tensor* reshape_3d(
        ggml_tensor* tensor,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2
    ) override;

    ggml_tensor* reshape_4d(
        ggml_tensor* tensor,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3
    ) override;

    // Permute / transpose
    ggml_tensor* permute(
        ggml_tensor* tensor,
        int axis0,
        int axis1,
        int axis2,
        int axis3
    ) override;

    // Views
    ggml_tensor* view_1d(
        ggml_tensor* tensor,
        int64_t ne0,
        size_t offset
    ) override;

    ggml_tensor* view_2d(
        ggml_tensor* tensor,
        int64_t ne0,
        int64_t ne1,
        size_t nb1,
        size_t offset
    ) override;

    ggml_tensor* view_3d(
        ggml_tensor* tensor,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        size_t nb1,
        size_t nb2,
        size_t offset
    ) override;

    ggml_tensor* view_4d(
        ggml_tensor* tensor,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3,
        size_t nb1,
        size_t nb2,
        size_t nb3,
        size_t offset
    ) override;

    // Repeat
    ggml_tensor* repeat(
        ggml_tensor* tensor,
        ggml_tensor* target
    ) override;

    // Concatenation
    ggml_tensor* concat(
        ggml_tensor* a,
        ggml_tensor* b,
        int dim
    ) override;

    // Reduction
    ggml_tensor* sum_rows(
        ggml_tensor* tensor
    ) override;

    ggml_tensor* silu(
        ggml_tensor* tensor
    ) override;
};
