#include "ggml/PlannerEngine.hpp"
#include "ggml/Scope.hpp"
#include "ggml/Context.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <iostream>

// -----------------------------------------------------------------------------
// Construction
// -----------------------------------------------------------------------------

PlannerEngine::PlannerEngine(
    int device_count,
    double communication_weight,
    double compute_weight,
    double memory_weight)
    : device_count_(device_count),
      communication_weight_(communication_weight),
      compute_weight_(compute_weight),
      memory_weight_(memory_weight)
{
    if (device_count_ <= 0)
        throw std::invalid_argument(
            "PlannerEngine: device_count must be greater than zero");
}


// -----------------------------------------------------------------------------
// Tensor registration
// -----------------------------------------------------------------------------

PlannerEngine::TensorId PlannerEngine::tensor(
    ggml_tensor* tensor,
    const std::string& name)
{
    if (!tensor)
        throw std::invalid_argument(
            "PlannerEngine::tensor(): null tensor");

    auto it = tensor_ids_.find(tensor);

    if (it != tensor_ids_.end())
        return it->second;

    Node n;

    n.info.id = next_tensor_id_++;
    n.info.type = tensor->type;
    n.info.n_dims = ggml_n_dims(tensor);
    n.info.bytes = ggml_nbytes(tensor);
    n.info.name = ggml_get_name(tensor);

    for (int i = 0; i < ggml_n_dims(tensor); ++i)
        n.info.ne[i] = tensor->ne[i];

    nodes_.push_back(std::move(n));

    tensor_ids_[tensor] = nodes_.back().info.id;

    return nodes_.back().info.id;
}

PlannerEngine::TensorId PlannerEngine::id_of(
    ggml_tensor* tensor) const
{
    if (!tensor)
        throw std::invalid_argument(
            "PlannerEngine::id_of(): null tensor");

    auto it = tensor_ids_.find(tensor);

    if (it == tensor_ids_.end())
        throw std::runtime_error(
            "PlannerEngine::id_of(): tensor is not registered");

    return it->second;
}


PlannerEngine::Node& PlannerEngine::node(
    TensorId id)
{
    if (id >= nodes_.size())
        throw std::out_of_range(
            "Planner: invalid TensorId");

    return nodes_[id];
}


const PlannerEngine::Node& PlannerEngine::node(
    TensorId id) const
{
    if (id >= nodes_.size())
        throw std::out_of_range(
            "Planner: invalid TensorId");

    return nodes_[id];
}


PlannerEngine::Operation& PlannerEngine::operation(
    OperationId id)
{
    for (auto& op : operations_) {
        if (op.id == id)
            return op;
    }

    throw std::out_of_range(
        "Planner: invalid OperationId");
}


const PlannerEngine::Operation& PlannerEngine::operation(
    OperationId id) const
{
    for (const auto& op : operations_) {
        if (op.id == id)
            return op;
    }

    throw std::out_of_range(
        "Planner: invalid OperationId");
}


// -----------------------------------------------------------------------------
// Symbolic tensor creation
// -----------------------------------------------------------------------------

ggml_tensor* PlannerEngine::create_output(
    ggml_context* ctx,
    ggml_type type,
    int n_dims,
    const int64_t* ne)
{
    /*
     * IMPORTANT:
     *
     * We create only a symbolic ggml tensor descriptor.
     *
     * We do NOT call ggml_add(), ggml_mul_mat(), etc.
     *
     * The returned tensor exists only so that the Tensor wrapper can
     * continue carrying shape/type information through the planning pass.
     */
    ggml_tensor* result =
        ggml_new_tensor(
            ctx,
            type,
            n_dims,
            ne);

    if (!result)
        throw std::runtime_error(
            "Planner: failed to create symbolic ggml tensor");

    tensor(result);

    return result;
}


ggml_tensor* PlannerEngine::new_tensor(
    ggml_context* ctx,
    ggml_type type,
    int n_dims,
    const int64_t* ne)
{
    return create_output(ctx, type, n_dims, ne);
}


ggml_tensor* PlannerEngine::new_tensor_1d(
    ggml_context* ctx,
    ggml_type type,
    int64_t ne0)
{
    const int64_t ne[] = { ne0 };

    return create_output(
        ctx,
        type,
        1,
        ne);
}


void PlannerEngine::set_input(
    ggml_tensor* tensor)
{
    const TensorId id = id_of(tensor);

    node(id).is_input = true;
}


// -----------------------------------------------------------------------------
// Operation recording
// -----------------------------------------------------------------------------

PlannerEngine::OperationId PlannerEngine::record_operation(
    const std::string& name,
    const std::vector<TensorId>& inputs,
    TensorId output,
    std::vector<Candidate> candidates)
{
    Operation op;

    op.id = next_operation_id_++;
    op.name = name;
    op.inputs = inputs;
    op.output = output;
    op.candidates = std::move(candidates);

    operations_.push_back(std::move(op));

    node(output).producer =
        operations_.back().id;

    return operations_.back().id;
}


// -----------------------------------------------------------------------------
// Candidate helpers
// -----------------------------------------------------------------------------

std::vector<PlannerEngine::Candidate>
PlannerEngine::unary_candidates(
    TensorId input,
    const char* name) const
{
    std::vector<Candidate> candidates;

    /*
     * R -> R
     */
    candidates.push_back({
        Distribution::replicated(),

        {
            { input, Distribution::replicated() }
        },

        0.0,
        0.0,
        0.0,

        std::string(name) + ": R -> R"
    });

    /*
     * S(a) -> S(a)
     */
    for (int axis = 0;
         axis < GGML_MAX_DIMS;
         ++axis)
    {
        candidates.push_back({
            Distribution::sharded(axis),

            {
                { input, Distribution::sharded(axis) }
            },

            0.0,
            0.0,
            0.0,

            std::string(name) +
                ": S(" +
                std::to_string(axis) +
                ") -> S(" +
                std::to_string(axis) +
                ")"
        });
    }

    /*
     * P(a) -> P(a)
     */
    for (int axis = 0;
         axis < GGML_MAX_DIMS;
         ++axis)
    {
        candidates.push_back({
            Distribution::partial(axis),

            {
                { input, Distribution::partial(axis) }
            },

            0.0,
            0.0,
            0.0,

            std::string(name) +
                ": P(" +
                std::to_string(axis) +
                ") -> P(" +
                std::to_string(axis) +
                ")"
        });
    }

    return candidates;
}


std::vector<PlannerEngine::Candidate>
PlannerEngine::elementwise_candidates(
    TensorId lhs,
    TensorId rhs,
    const char* name) const
{
    std::vector<Candidate> candidates;

    /*
     * R + R -> R
     */
    candidates.push_back({
        Distribution::replicated(),

        {
            { lhs, Distribution::replicated() },
            { rhs, Distribution::replicated() }
        },

        0.0,
        0.0,
        0.0,

        std::string(name) + ": R,R -> R"
    });

    for (int axis = 0;
         axis < GGML_MAX_DIMS;
         ++axis)
    {
        /*
         * R + S(a) -> S(a)
         */
        candidates.push_back({
            Distribution::sharded(axis),

            {
                { lhs, Distribution::replicated() },
                { rhs, Distribution::sharded(axis) }
            },

            0.0,
            0.0,
            0.0,

            std::string(name) +
                ": R,S(" +
                std::to_string(axis) +
                ") -> S(" +
                std::to_string(axis) +
                ")"
        });

        /*
         * S(a) + R -> S(a)
         */
        candidates.push_back({
            Distribution::sharded(axis),

            {
                { lhs, Distribution::sharded(axis) },
                { rhs, Distribution::replicated() }
            },

            0.0,
            0.0,
            0.0,

            std::string(name) +
                ": S(" +
                std::to_string(axis) +
                "),R -> S(" +
                std::to_string(axis) +
                ")"
        });

        /*
         * S(a) + S(a) -> S(a)
         */
        candidates.push_back({
            Distribution::sharded(axis),

            {
                { lhs, Distribution::sharded(axis) },
                { rhs, Distribution::sharded(axis) }
            },

            0.0,
            0.0,
            0.0,

            std::string(name) +
                ": S(" +
                std::to_string(axis) +
                "),S(" +
                std::to_string(axis) +
                ") -> S(" +
                std::to_string(axis) +
                ")"
        });

        /*
         * P(a) + P(a) -> P(a)
         */
        candidates.push_back({
            Distribution::partial(axis),

            {
                { lhs, Distribution::partial(axis) },
                { rhs, Distribution::partial(axis) }
            },

            0.0,
            0.0,
            0.0,

            std::string(name) +
                ": P(" +
                std::to_string(axis) +
                "),P(" +
                std::to_string(axis) +
                ") -> P(" +
                std::to_string(axis) +
                ")"
        });
    }

    return candidates;
}


// -----------------------------------------------------------------------------
// Generic unary operation
// -----------------------------------------------------------------------------

ggml_tensor* PlannerEngine::unary(
    const char* name,
    ggml_tensor* input,
    ggml_type output_type,
    const int64_t* output_ne,
    int n_dims)
{
    const TensorId input_id =
        id_of(input);

    ggml_tensor* output =
        create_output(
            *Scope::context(),
            output_type,
            n_dims,
            output_ne);

    const TensorId output_id =
        id_of(output);

    record_operation(
        name,
        { input_id },
        output_id,
        unary_candidates(
            input_id,
            name));

    return output;
}


// -----------------------------------------------------------------------------
// Generic binary elementwise operation
// -----------------------------------------------------------------------------

ggml_tensor* PlannerEngine::binary(
    const char* name,
    ggml_tensor* lhs,
    ggml_tensor* rhs)
{
    const TensorId lhs_id =
        id_of(lhs);

    const TensorId rhs_id =
        id_of(rhs);

    const int n_dims =
        std::max(
            ggml_n_dims(lhs),
            ggml_n_dims(rhs));

    int64_t ne[GGML_MAX_DIMS] = {};

    for (int i = 0; i < n_dims; ++i) {
        const int64_t lhs_dim =
            i < ggml_n_dims(lhs)
                ? lhs->ne[i]
                : 1;

        const int64_t rhs_dim =
            i < ggml_n_dims(rhs)
                ? rhs->ne[i]
                : 1;

        ne[i] =
            std::max(lhs_dim, rhs_dim);
    }

    ggml_tensor* output =
        create_output(
            *Scope::context(),
            lhs->type,
            n_dims,
            ne);

    const TensorId output_id =
        id_of(output);

    record_operation(
        name,
        { lhs_id, rhs_id },
        output_id,
        elementwise_candidates(
            lhs_id,
            rhs_id,
            name));

    return output;
}


// -----------------------------------------------------------------------------
// Primitive implementations
// -----------------------------------------------------------------------------

ggml_tensor* PlannerEngine::fill(
    ggml_tensor* a,
    float)
{
    return unary(
        "fill",
        a,
        a->type,
        a->ne,
        ggml_n_dims(a));
}


ggml_tensor* PlannerEngine::cont(
    ggml_tensor* a)
{
    return unary(
        "cont",
        a,
        a->type,
        a->ne,
        ggml_n_dims(a));
}


ggml_tensor* PlannerEngine::dup(
    ggml_tensor* a)
{
    return unary(
        "dup",
        a,
        a->type,
        a->ne,
        ggml_n_dims(a));
}


ggml_tensor* PlannerEngine::cast(
    ggml_tensor* a,
    ggml_type type)
{
    return unary(
        "cast",
        a,
        type,
        a->ne,
        ggml_n_dims(a));
}


ggml_tensor* PlannerEngine::cpy(
    ggml_tensor* a,
    ggml_tensor* b)
{
    return unary(
        "cpy",
        a,
        b->type,
        b->ne,
        ggml_n_dims(b));
}


ggml_tensor* PlannerEngine::neg(
    ggml_tensor* a)
{
    return unary(
        "neg",
        a,
        a->type,
        a->ne,
        ggml_n_dims(a));
}


ggml_tensor* PlannerEngine::abs(
    ggml_tensor* a)
{
    return unary(
        "abs",
        a,
        a->type,
        a->ne,
        ggml_n_dims(a));
}


ggml_tensor* PlannerEngine::sqrt(
    ggml_tensor* a)
{
    return unary(
        "sqrt",
        a,
        a->type,
        a->ne,
        ggml_n_dims(a));
}


ggml_tensor* PlannerEngine::exp(
    ggml_tensor* a)
{
    return unary(
        "exp",
        a,
        a->type,
        a->ne,
        ggml_n_dims(a));
}


ggml_tensor* PlannerEngine::log(
    ggml_tensor* a)
{
    return unary(
        "log",
        a,
        a->type,
        a->ne,
        ggml_n_dims(a));
}


ggml_tensor* PlannerEngine::sin(
    ggml_tensor* a)
{
    return unary(
        "sin",
        a,
        a->type,
        a->ne,
        ggml_n_dims(a));
}


ggml_tensor* PlannerEngine::cos(
    ggml_tensor* a)
{
    return unary(
        "cos",
        a,
        a->type,
        a->ne,
        ggml_n_dims(a));
}


ggml_tensor* PlannerEngine::add(
    ggml_tensor* a,
    ggml_tensor* b)
{
    return binary(
        "add",
        a,
        b);
}


ggml_tensor* PlannerEngine::sub(
    ggml_tensor* a,
    ggml_tensor* b)
{
    return binary(
        "sub",
        a,
        b);
}


ggml_tensor* PlannerEngine::mul(
    ggml_tensor* a,
    ggml_tensor* b)
{
    return binary(
        "mul",
        a,
        b);
}


ggml_tensor* PlannerEngine::div(
    ggml_tensor* a,
    ggml_tensor* b)
{
    return binary(
        "div",
        a,
        b);
}


ggml_tensor* PlannerEngine::scale(
    ggml_tensor* a,
    float)
{
    return unary(
        "scale",
        a,
        a->type,
        a->ne,
        ggml_n_dims(a));
}


ggml_tensor* PlannerEngine::clamp(
    ggml_tensor* a,
    float,
    float)
{
    return unary(
        "clamp",
        a,
        a->type,
        a->ne,
        ggml_n_dims(a));
}


ggml_tensor* PlannerEngine::mul_mat(
    ggml_tensor* a,
    ggml_tensor* b)
{
    if (!a || !b)
        throw std::invalid_argument(
            "Planner::mul_mat(): null operand");

    /*
     * GGML's logical matrix multiplication:
     *
     *     result = A^T * B
     *
     * In the PyTorch-like wrapper, Linear weights are represented
     * logically as:
     *
     *     W = [out_features, in_features]
     *
     * while GGML stores the dimensions reversed:
     *
     *     W = [in_features, out_features]
     *
     * Therefore the distribution axes below are expressed in the
     * wrapper's logical tensor coordinates, not GGML's physical
     * dimension numbering.
     *
     * For a Linear:
     *
     *     weight: [out, in]
     *     input:  [..., in]
     *     output: [..., out]
     *
     * Candidate strategies:
     *
     *   1. Replicated:
     *
     *        input  R
     *        weight R
     *        output R
     *
     *   2. Column parallel:
     *
     *        input  R
     *        weight S(output)
     *        output S(output)
     *
     *   3. Row parallel:
     *
     *        input  S(input)
     *        weight S(input)
     *        output P(output)
     *
     * The planner doesn't know that 'b' is a Linear weight.
     * It only knows that this is matrix multiplication.
     */

    const TensorId a_id = id_of(a);
    const TensorId b_id = id_of(b);

    /*
     * -------------------------------------------------------------------------
     * Determine output shape.
     * -------------------------------------------------------------------------
     *
     * For GGML:
     *
     *     A: [K, M]
     *     B: [K, N]
     *     result: [N, M]
     *
     * For the common Linear case:
     *
     *     weight: [in, out]  -> GGML A
     *     x:      [in, ...]  -> GGML B
     *
     * result:
     *
     *     [out, ...]
     *
     * ggml_mul_mat() preserves the higher dimensions of B.
     */

    if (ggml_n_dims(a) < 2 || ggml_n_dims(b) < 1)
        throw std::invalid_argument(
            "Planner::mul_mat(): invalid operand ranks");

    if (a->ne[0] != b->ne[0])
        throw std::invalid_argument(
            "Planner::mul_mat(): incompatible contraction dimensions");

    int64_t ne[GGML_MAX_DIMS] = {};

    /*
     * Result dimension 0 comes from A dimension 1.
     */
    ne[0] = a->ne[1];

    /*
     * Remaining dimensions come from B.
     *
     * This follows ggml_mul_mat()'s normal output convention.
     */
    for (int i = 1; i < ggml_n_dims(b); ++i)
        ne[i] = b->ne[i];

    const int n_dims =
        std::max(1, ggml_n_dims(b));

    ggml_tensor* output =
        create_output(
            *Scope::context(),
            a->type,
            n_dims,
            ne);

    const TensorId output_id =
        id_of(output);

    std::vector<Candidate> candidates;

    /*
     * -------------------------------------------------------------------------
     * Candidate 1: fully replicated
     * -------------------------------------------------------------------------
     *
     *       A R
     *       B R
     *       ----
     *       O R
     *
     * This is always the safe baseline.
     */
    candidates.push_back({
        Distribution::replicated(),

        {
            { a_id, Distribution::replicated() },
            { b_id, Distribution::replicated() }
        },

        0.0,
        0.0,
        0.0,

        "mul_mat: R,R -> R"
    });

    /*
     * -------------------------------------------------------------------------
     * Candidate 2: column parallel
     * -------------------------------------------------------------------------
     *
     * Logical:
     *
     *       W = [out, in]
     *
     * split W along output dimension:
     *
     *       W_i = [out_i, in]
     *
     * Therefore:
     *
     *       input  = R
     *       weight = S(output)
     *       output = S(output)
     *
     * In this generic operator representation, axis 0 is the logical
     * output-feature axis.
     *
     * The module layer can later map this requirement to the physical
     * GGML axis when resolving the Parameter.
     */
    candidates.push_back({
        Distribution::sharded(0),

        {
            { a_id, Distribution::replicated() },
            { b_id, Distribution::sharded(0) }
        },

        0.0,
        0.0,
        0.0,

        "mul_mat: R,S(output) -> S(output)"
    });

    /*
     * -------------------------------------------------------------------------
     * Candidate 3: row parallel
     * -------------------------------------------------------------------------
     *
     * Logical:
     *
     *       W = [out, in]
     *
     * split W along input dimension:
     *
     *       W_i = [out, in_i]
     *
     * and split the activation accordingly:
     *
     *       X_i = [..., in_i]
     *
     * Each GPU computes:
     *
     *       Y_i = W_i X_i
     *
     * The mathematical result is:
     *
     *       Y = sum_i Y_i
     *
     * so the immediate output is Partial.
     *
     *       input  = S(input)
     *       weight = S(input)
     *       output = P(output)
     *
     * For Linear's usual [out,in] logical representation:
     *
     *       weight shard axis = 1
     *       activation shard axis = last/input feature axis
     *
     * We represent the output as P(0), because the partial result
     * corresponds to the output tensor.
     */
    candidates.push_back({
        Distribution::partial(0),

        {
            { a_id, Distribution::sharded(1) },
            { b_id, Distribution::sharded(1) }
        },

        0.0,
        0.0,
        0.0,

        "mul_mat: S(input),S(input) -> P(output)"
    });

    /*
     * -------------------------------------------------------------------------
     * Additional batch/sequence parallel candidates
     * -------------------------------------------------------------------------
     *
     * If the second operand is itself batch/sequence distributed, a
     * matrix multiplication can preserve that distribution.
     *
     * For example:
     *
     *       A = replicated weight
     *       B = S(batch)
     *       O = S(batch)
     *
     * This is particularly useful when the activation tensor is already
     * sequence/batch sharded.
     *
     * We expose these as generic candidates. Whether a particular one is
     * mathematically/physically valid for a given rank/layout should be
     * tightened later using operand shape metadata.
     */
    for (int axis = 2;
         axis < GGML_MAX_DIMS;
         ++axis)
    {
        candidates.push_back({
            Distribution::sharded(axis),

            {
                { a_id, Distribution::replicated() },
                { b_id, Distribution::sharded(axis) }
            },

            0.0,
            0.0,
            0.0,

            "mul_mat: R,S(" +
                std::to_string(axis) +
                ") -> S(" +
                std::to_string(axis) +
                ")"
        });
    }

    record_operation(
        "mul_mat",
        { a_id, b_id },
        output_id,
        std::move(candidates));

    return output;
}

// -----------------------------------------------------------------------------
// Reshape
// -----------------------------------------------------------------------------

ggml_tensor* PlannerEngine::reshape_1d(
    ggml_tensor* a,
    int64_t ne0)
{
    const int64_t ne[] = { ne0 };

    return unary(
        "reshape_1d",
        a,
        a->type,
        ne,
        1);
}


ggml_tensor* PlannerEngine::reshape_2d(
    ggml_tensor* a,
    int64_t ne0,
    int64_t ne1)
{
    const int64_t ne[] = {
        ne0,
        ne1
    };

    return unary(
        "reshape_2d",
        a,
        a->type,
        ne,
        2);
}


ggml_tensor* PlannerEngine::reshape_3d(
    ggml_tensor* a,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2)
{
    const int64_t ne[] = {
        ne0,
        ne1,
        ne2
    };

    return unary(
        "reshape_3d",
        a,
        a->type,
        ne,
        3);
}


ggml_tensor* PlannerEngine::reshape_4d(
    ggml_tensor* a,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2,
    int64_t ne3)
{
    const int64_t ne[] = {
        ne0,
        ne1,
        ne2,
        ne3
    };

    return unary(
        "reshape_4d",
        a,
        a->type,
        ne,
        4);
}


// -----------------------------------------------------------------------------
// Permute
// -----------------------------------------------------------------------------

ggml_tensor* PlannerEngine::permute(
    ggml_tensor* a,
    int axis0,
    int axis1,
    int axis2,
    int axis3)
{
    const int axes[] = {
        axis0,
        axis1,
        axis2,
        axis3
    };

    int64_t ne[GGML_MAX_DIMS] = {};

    for (int i = 0; i < ggml_n_dims(a); ++i)
        ne[i] = a->ne[axes[i]];

    return unary(
        "permute",
        a,
        a->type,
        ne,
        ggml_n_dims(a));
}


// -----------------------------------------------------------------------------
// Views
// -----------------------------------------------------------------------------

ggml_tensor* PlannerEngine::view_1d(
    ggml_tensor* a,
    int64_t ne0,
    size_t)
{
    const int64_t ne[] = { ne0 };

    return unary(
        "view_1d",
        a,
        a->type,
        ne,
        1);
}


ggml_tensor* PlannerEngine::view_2d(
    ggml_tensor* a,
    int64_t ne0,
    int64_t ne1,
    size_t,
    size_t)
{
    const int64_t ne[] = {
        ne0,
        ne1
    };

    return unary(
        "view_2d",
        a,
        a->type,
        ne,
        2);
}


ggml_tensor* PlannerEngine::view_3d(
    ggml_tensor* a,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2,
    size_t,
    size_t,
    size_t)
{
    const int64_t ne[] = {
        ne0,
        ne1,
        ne2
    };

    return unary(
        "view_3d",
        a,
        a->type,
        ne,
        3);
}


ggml_tensor* PlannerEngine::view_4d(
    ggml_tensor* a,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2,
    int64_t ne3,
    size_t,
    size_t,
    size_t,
    size_t)
{
    const int64_t ne[] = {
        ne0,
        ne1,
        ne2,
        ne3
    };

    return unary(
        "view_4d",
        a,
        a->type,
        ne,
        4);
}


// -----------------------------------------------------------------------------
// Repeat
// -----------------------------------------------------------------------------

ggml_tensor* PlannerEngine::repeat(
    ggml_tensor* a,
    ggml_tensor* b)
{
    return unary(
        "repeat",
        a,
        a->type,
        b->ne,
        ggml_n_dims(b));
}


// -----------------------------------------------------------------------------
// Concat
// -----------------------------------------------------------------------------

ggml_tensor* PlannerEngine::concat(
    ggml_tensor* a,
    ggml_tensor* b,
    int dim)
{
    if (ggml_n_dims(a) != ggml_n_dims(b))
        throw std::invalid_argument(
            "PlannerEngine::concat(): rank mismatch");

    int64_t ne[GGML_MAX_DIMS] = {};

    for (int i = 0; i < ggml_n_dims(a); ++i)
        ne[i] = a->ne[i];

    /*
     * Tensor::cat() has already converted the logical concatenation
     * dimension into ggml's dimension 0 before reaching this method.
     */
    ne[0] =
        a->ne[0] + b->ne[0];

    ggml_tensor* output =
        create_output(
            *Scope::context(),
            a->type,
            ggml_n_dims(a),
            ne);

    const TensorId lhs =
        id_of(a);

    const TensorId rhs =
        id_of(b);

    const TensorId out =
        id_of(output);

    /*
     * Concatenation can preserve a sharding layout when the
     * concatenation dimension itself is sharded.
     *
     * For now we expose the safe generic choices.
     */
    std::vector<Candidate> candidates;

    candidates.push_back({
        Distribution::replicated(),

        {
            { lhs, Distribution::replicated() },
            { rhs, Distribution::replicated() }
        },

        0.0,
        0.0,
        0.0,

        "concat: R,R -> R"
    });

    for (int axis = 0;
         axis < GGML_MAX_DIMS;
         ++axis)
    {
        candidates.push_back({
            Distribution::sharded(axis),

            {
                { lhs, Distribution::sharded(axis) },
                { rhs, Distribution::sharded(axis) }
            },

            0.0,
            0.0,
            0.0,

            "concat: S(" +
                std::to_string(axis) +
                "),S(" +
                std::to_string(axis) +
                ") -> S(" +
                std::to_string(axis) +
                ")"
        });
    }

    record_operation(
        "concat",
        { lhs, rhs },
        out,
        std::move(candidates));

    return output;
}


// -----------------------------------------------------------------------------
// Sum rows
// -----------------------------------------------------------------------------

ggml_tensor* PlannerEngine::sum_rows(
    ggml_tensor* a)
{
    int64_t ne[GGML_MAX_DIMS] = {};

    /*
     * ggml_sum_rows reduces ggml dimension 0.
     */
    ne[0] = 1;

    for (int i = 1; i < ggml_n_dims(a); ++i)
        ne[i] = a->ne[i];

    ggml_tensor* output =
        create_output(
            *Scope::context(),
            a->type,
            ggml_n_dims(a),
            ne);

    const TensorId input =
        id_of(a);

    const TensorId out =
        id_of(output);

    std::vector<Candidate> candidates;

    /*
     * R -> R
     */
    candidates.push_back({
        Distribution::replicated(),

        {
            { input, Distribution::replicated() }
        },

        0.0,
        0.0,
        0.0,

        "sum_rows: R -> R"
    });

    /*
     * If the reduced dimension is sharded, the result is partial.
     *
     * ggml dimension 0 corresponds to logical dimension determined
     * by the Tensor wrapper.
     */
    candidates.push_back({
        Distribution::partial(0),

        {
            { input, Distribution::sharded(0) }
        },

        0.0,
        0.0,
        0.0,

        "sum_rows: S(0) -> P(0)"
    });

    /*
     * If another dimension is sharded, reduction is local and the
     * sharding can remain.
     */
    for (int axis = 1;
         axis < GGML_MAX_DIMS;
         ++axis)
    {
        candidates.push_back({
            Distribution::sharded(axis),

            {
                { input, Distribution::sharded(axis) }
            },

            0.0,
            0.0,
            0.0,

            "sum_rows: S(" +
                std::to_string(axis) +
                ") -> S(" +
                std::to_string(axis) +
                ")"
        });
    }

    record_operation(
        "sum_rows",
        { input },
        out,
        std::move(candidates));

    return output;
}

ggml_tensor * PlannerEngine::silu(ggml_tensor * a) {
    if (!a) {
        throw std::invalid_argument(
            "Planner::silu(): null operand");
    }

    const TensorId a_id = id_of(a);

    // SiLU is elementwise:
    //
    //     silu(x) = x * sigmoid(x)
    //
    // but from the planner's point of view it is one ggml
    // operation, corresponding directly to ggml_silu().
    //
    // Therefore the distribution is preserved:
    //
    //     R      -> R
    //     S(a)   -> S(a)
    //     P(a)   -> P(a)

    ggml_tensor * output =
        create_output(
            *Scope::context(),
            a->type,
            ggml_n_dims(a),
            a->ne);

    const TensorId output_id = id_of(output);

    std::vector<Candidate> candidates;

    // R -> R
    candidates.push_back({
        Distribution::replicated(),

        {
            { a_id, Distribution::replicated() }
        },

        0.0, // communication
        0.0, // compute
        0.0, // memory

        "silu: R -> R"
    });

    // S(axis) -> S(axis)
    //
    // SiLU operates independently on every element, so whatever
    // shard axis the input has, the output can keep it.
    for (int axis = 0; axis < GGML_MAX_DIMS; ++axis) {
        candidates.push_back({
            Distribution::sharded(axis),

            {
                { a_id, Distribution::sharded(axis) }
            },

            0.0,
            0.0,
            0.0,

            "silu: S(" + std::to_string(axis) +
                ") -> S(" + std::to_string(axis) + ")"
        });
    }

    // P(axis) -> P(axis)
    //
    // A partial tensor can also be transformed elementwise without
    // changing its distribution semantics.
    for (int axis = 0; axis < GGML_MAX_DIMS; ++axis) {
        candidates.push_back({
            Distribution::partial(axis),

            {
                { a_id, Distribution::partial(axis) }
            },

            0.0,
            0.0,
            0.0,

            "silu: P(" + std::to_string(axis) +
                ") -> P(" + std::to_string(axis) + ")"
        });
    }

    record_operation(
        "silu",
        { a_id },
        output_id,
        std::move(candidates));

    return output;
}

// -----------------------------------------------------------------------------
// Communication cost
// -----------------------------------------------------------------------------

double PlannerEngine::communication_cost(
    const Distribution& from,
    const Distribution& to,
    const TensorInfo& tensor) const
{
    if (from == to)
        return 0.0;

    const double bytes =
        static_cast<double>(tensor.bytes);

    /*
     * This is intentionally a simple topology-independent model.
     *
     * Replace this function later with:
     *
     *     topology_.cost(from, to, bytes)
     *
     * so PCIe/NVLink/P2P/asymmetric links can be modeled.
     */

    if (from.is_replicated() &&
        to.is_sharded())
    {
        /*
         * Scatter.
         */
        return bytes;
    }

    if (from.is_sharded() &&
        to.is_replicated())
    {
        /*
         * All-gather.
         */
        return bytes;
    }

    if (from.is_partial() &&
        to.is_replicated())
    {
        /*
         * All-reduce.
         */
        return bytes;
    }

    if (from.is_replicated() &&
        to.is_partial())
    {
        return bytes;
    }

    if (from.is_sharded() &&
        to.is_partial())
    {
        /*
         * The local operation can naturally produce a partial
         * result, so no communication is required merely to change
         * the semantic state.
         */
        return 0.0;
    }

    if (from.is_partial() &&
        to.is_sharded())
    {
        /*
         * Reduce-scatter.
         */
        return bytes;
    }

    if (from.is_sharded() &&
        to.is_sharded())
    {
        /*
         * Redistribution between shard axes.
         */
        return bytes;
    }

    if (from.is_partial() &&
        to.is_partial())
    {
        return bytes;
    }

    return bytes;
}


// -----------------------------------------------------------------------------
// Compute cost
// -----------------------------------------------------------------------------

double PlannerEngine::distribution_compute_cost(
    const Distribution& distribution,
    const TensorInfo& tensor) const
{
    const double work =
        static_cast<double>(tensor.bytes);

    /*
     * Replicated means every GPU performs the operation.
     */
    if (distribution.is_replicated())
        return work *
               static_cast<double>(device_count_);

    /*
     * Sharded / partial means work is distributed.
     *
     * This is the idealized equal-device case.
     *
     * Later this should become a weighted model based on the actual
     * GPUs, e.g.:
     *
     *     4060 Ti = 1.0
     *     3060    = 0.75
     *
     * and calculate the optimal shard boundaries accordingly.
     */
    if (distribution.is_sharded() ||
        distribution.is_partial())
        return work;

    return work;
}


// -----------------------------------------------------------------------------
// Candidate cost
// -----------------------------------------------------------------------------

double PlannerEngine::candidate_local_cost(
    const Operation& op,
    const Candidate& candidate) const
{
    const TensorInfo& output =
        node(op.output).info;

    const double compute =
        candidate.compute_cost != 0.0
            ? candidate.compute_cost
            : distribution_compute_cost(
                  candidate.output,
                  output);

    return
        communication_weight_ *
            candidate.communication_cost
        +
        compute_weight_ *
            compute
        +
        memory_weight_ *
            candidate.memory_cost;
}


// -----------------------------------------------------------------------------
// DP helper
// -----------------------------------------------------------------------------

PlannerEngine::DPState
PlannerEngine::best_state_for_operand(
    TensorId tensor,
    const Distribution& required,
    const std::unordered_map<
        TensorId,
        std::unordered_map<
            Distribution,
            DPState,
            DistributionHash>>& states) const
{
    const Node& n =
        node(tensor);

    /*
     * Inputs have a known starting distribution.
     */
    if (n.is_input) {
        DPState result;

        result.cost =
            communication_cost(
                n.initial_distribution,
                required,
                n.info);

        result.producer =
            InvalidOperation;

        result.candidate =
            std::numeric_limits<size_t>::max();

        result.distribution =
            n.initial_distribution;

        return result;
    }

    auto states_it =
        states.find(tensor);

    if (states_it == states.end())
        throw std::runtime_error(
            "Planner DP: no states for tensor");

    DPState best;

    for (const auto& [produced, state] :
         states_it->second)
    {
        const double transition =
            communication_cost(
                produced,
                required,
                n.info);

        const double total =
            state.cost + transition;

        if (total < best.cost) {
            best = state;

            best.cost = total;

            /*
             * The distribution stored here is the distribution
             * actually produced by the operand's producer.
             */
            best.distribution =
                produced;
        }
    }

    return best;
}


// -----------------------------------------------------------------------------
// DP
// -----------------------------------------------------------------------------

void PlannerEngine::solve_operation(
    const Operation& op,
    std::unordered_map<
        TensorId,
        std::unordered_map<
            Distribution,
            DPState,
            DistributionHash>>& states)
{
    auto& output_states =
        states[op.output];

    for (size_t candidate_index = 0;
         candidate_index < op.candidates.size();
         ++candidate_index)
    {
        const Candidate& candidate =
            op.candidates[candidate_index];

        double cost =
            candidate_local_cost(
                op,
                candidate);

        bool valid = true;

        /*
         * For every operand, find the cheapest already-computed
         * distribution from which the required distribution can be
         * obtained.
         */
        for (const OperandRequirement& requirement :
             candidate.operands)
        {
            DPState operand =
                best_state_for_operand(
                    requirement.tensor,
                    requirement.distribution,
                    states);

            if (!std::isfinite(operand.cost)) {
                valid = false;
                break;
            }

            cost += operand.cost;
        }

        if (!valid)
            continue;

        auto it =
            output_states.find(
                candidate.output);

        if (it != output_states.end() &&
            it->second.cost <= cost)
        {
            continue;
        }

        DPState state;

        state.cost = cost;

        state.producer =
            op.id;

        state.candidate =
            candidate_index;

        state.distribution =
            candidate.output;

        output_states[
            candidate.output] =
                std::move(state);
    }
}


// -----------------------------------------------------------------------------
// Solve entire graph
// -----------------------------------------------------------------------------

void PlannerEngine::solve()
{
    using StateMap =
        std::unordered_map<
            TensorId,
            std::unordered_map<
                Distribution,
                DPState,
                DistributionHash>>;

    StateMap states;

    /*
     * -------------------------------------------------------------------------
     * 1. Initialize graph inputs.
     * -------------------------------------------------------------------------
     */

    for (const Node& n : nodes_) {
        if (!n.is_input)
            continue;

        DPState state;

        state.cost = 0.0;

        state.producer =
            InvalidOperation;

        state.candidate =
            std::numeric_limits<size_t>::max();

        state.distribution =
            n.initial_distribution;

        states[n.info.id][
            n.initial_distribution] =
                state;
    }

    /*
     * -------------------------------------------------------------------------
     * 2. Forward dynamic programming.
     * -------------------------------------------------------------------------
     *
     * Operations are recorded during forward execution, therefore
     * operations_ are already topologically ordered.
     */
    for (const Operation& op : operations_)
        solve_operation(op, states);

    /*
     * -------------------------------------------------------------------------
     * 3. Find graph outputs.
     * -------------------------------------------------------------------------
     *
     * For now we treat the final operation output as the graph output.
     *
     * Once your Scope explicitly registers outputs, replace this with
     * the registered output set.
     */
    if (operations_.empty()) {
        plan_ =
            std::make_unique<ExecutionPlan>();

        plan_->total_cost = 0.0;

        return;
    }

    const Operation& final_operation =
        operations_.back();

    auto final_it =
        states.find(final_operation.output);

    if (final_it == states.end() ||
        final_it->second.empty())
    {
        throw std::runtime_error(
            "Planner: graph has no valid execution plan");
    }

    /*
     * Find cheapest final distribution.
     */
    const DPState* best_final = nullptr;

    for (const auto& [distribution, state] :
         final_it->second)
    {
        if (!best_final ||
            state.cost < best_final->cost)
        {
            best_final = &state;
        }
    }

    if (!best_final)
        throw std::runtime_error(
            "Planner: failed to select final state");

    /*
     * -------------------------------------------------------------------------
     * 4. Build plan.
     * -------------------------------------------------------------------------
     */

    plan_ =
        std::make_unique<ExecutionPlan>();

    plan_->total_cost =
        best_final->cost;

    /*
     * -------------------------------------------------------------------------
     * 5. Reconstruct the selected operation states.
     * -------------------------------------------------------------------------
     *
     * Because operations form a DAG rather than a simple chain,
     * reconstruct every operation whose output participates in the
     * selected path.
     *
     * The selected state is propagated backwards through the
     * candidate's operand requirements.
     */
    struct BacktrackState {
        TensorId tensor;
        Distribution distribution;
    };

    std::vector<BacktrackState> stack;

    stack.push_back({
        final_operation.output,
        best_final->distribution
    });

    std::unordered_map<
        TensorId,
        Distribution> visited;

    while (!stack.empty()) {
        BacktrackState current =
            stack.back();

        stack.pop_back();

        auto visited_it =
            visited.find(current.tensor);

        if (visited_it != visited.end()) {
            /*
             * If the same tensor is reached with the same distribution,
             * nothing else is required.
             */
            if (visited_it->second ==
                current.distribution)
            {
                continue;
            }
        }

        visited[current.tensor] =
            current.distribution;

        plan_->distributions[
            current.tensor] =
                current.distribution;

        const Node& n =
            node(current.tensor);

        /*
         * Input tensor.
         */
        if (n.is_input)
            continue;

        if (n.producer ==
            InvalidOperation)
        {
            throw std::runtime_error(
                "Planner: non-input tensor has no producer");
        }

        const Operation& op =
            operation(n.producer);

        const auto states_it =
            states.find(current.tensor);

        if (states_it == states.end())
            throw std::runtime_error(
                "Planner: missing DP state during backtracking");

        const auto state_it =
            states_it->second.find(
                current.distribution);

        if (state_it == states_it->second.end())
            throw std::runtime_error(
                "Planner: selected distribution has no DP state");

        const DPState& state =
            state_it->second;

        if (state.candidate >=
            op.candidates.size())
        {
            throw std::runtime_error(
                "Planner: invalid selected candidate");
        }

        const Candidate& candidate =
            op.candidates[state.candidate];

        SelectedCandidate selected;

        selected.operation =
            op.id;

        selected.candidate =
            state.candidate;

        selected.output =
            candidate.output;

        selected.operands =
            candidate.operands;

        selected.local_cost =
            candidate_local_cost(
                op,
                candidate);

        selected.accumulated_cost =
            state.cost;

        plan_->operations[
            op.id] =
                std::move(selected);

        /*
         * Continue backwards through every operand.
         */
        for (const OperandRequirement& requirement :
             candidate.operands)
        {
            stack.push_back({
                requirement.tensor,
                requirement.distribution
            });
        }
    }
}


// -----------------------------------------------------------------------------
// Finalize
// -----------------------------------------------------------------------------

void PlannerEngine::finalize()
{
    solve();
}

void PlannerEngine::print_plan(std::ostream& os) const
{
    if (!plan_)
        throw std::runtime_error(
            "Planner has not been finalized");

    auto tensor_name = [this](TensorId id) -> std::string {
        const auto& n = node(id);

        if (!n.info.name.empty())
            return n.info.name;

        return "tensor[" + std::to_string(id) + "]";
    };

    auto print_distribution =
        [&os](const Distribution& distribution) {
            os << distribution.to_string();
        };

    os << "\n";
    os << "================ Distribution Plan ================\n";
    os << "\n";

    /*
     * Inputs
     */
    os << "Inputs\n";
    os << "------\n";

    bool have_inputs = false;

    for (const auto& n : nodes_) {
        if (!n.is_input)
            continue;

        have_inputs = true;

        os << "  " << tensor_name(n.info.id)
           << " : ";

        auto it = plan_->distributions.find(n.info.id);

        if (it != plan_->distributions.end())
            print_distribution(it->second);
        else
            print_distribution(n.initial_distribution);

        os << "\n";
    }

    if (!have_inputs)
        os << "  <none>\n";

    os << "\n";

    /*
     * Operations
     */
    os << "Operations\n";
    os << "----------\n";

    for (const auto& op : operations_) {
        os << "\n";
        os << "[" << op.id << "] "
           << op.name << "\n";

        os << "  inputs:\n";

        for (TensorId input : op.inputs) {
            os << "    "
               << tensor_name(input)
               << " : ";

            auto it = plan_->distributions.find(input);

            if (it != plan_->distributions.end())
                print_distribution(it->second);
            else
                os << "<unresolved>";

            os << "\n";
        }

        os << "  output:\n";
        os << "    "
           << tensor_name(op.output)
           << " : ";

        auto output_it =
            plan_->distributions.find(op.output);

        if (output_it != plan_->distributions.end())
            print_distribution(output_it->second);
        else
            os << "<unresolved>";

        os << "\n";

        /*
         * Candidate list.
         */
        os << "  candidates:\n";

        for (size_t i = 0; i < op.candidates.size(); ++i) {
            const auto& candidate = op.candidates[i];

            const double local_cost =
                candidate_local_cost(op, candidate);

            os << "    [" << i << "] ";

            if (!candidate.description.empty())
                os << candidate.description << " : ";

            os << "output=";
            print_distribution(candidate.output);

            os << ", local_cost=" << local_cost;

            if (candidate.communication_cost != 0.0) {
                os << ", communication="
                   << candidate.communication_cost;
            }

            if (candidate.compute_cost != 0.0) {
                os << ", compute="
                   << candidate.compute_cost;
            }

            if (candidate.memory_cost != 0.0) {
                os << ", memory="
                   << candidate.memory_cost;
            }

            os << "\n";

            if (!candidate.operands.empty()) {
                os << "        operands:\n";

                for (const auto& operand :
                     candidate.operands) {

                    os << "          "
                       << tensor_name(operand.tensor)
                       << " : ";

                    print_distribution(
                        operand.distribution);

                    os << "\n";
                }
            }
        }

        /*
         * Selected candidate.
         */
        auto selected_it =
            plan_->operations.find(op.id);

        if (selected_it == plan_->operations.end()) {
            os << "  selected: <none>\n";
            continue;
        }

        const auto& selected = selected_it->second;

        os << "  selected:\n";
        os << "    candidate = "
           << selected.candidate
           << "\n";

        os << "    output = ";
        print_distribution(selected.output);
        os << "\n";

        if (!selected.operands.empty()) {
            os << "    operands:\n";

            for (const auto& operand :
                 selected.operands) {

                os << "      "
                   << tensor_name(operand.tensor)
                   << " : ";

                print_distribution(
                    operand.distribution);

                os << "\n";
            }
        }

        os << "    local_cost = "
           << selected.local_cost
           << "\n";

        os << "    accumulated_cost = "
           << selected.accumulated_cost
           << "\n";
    }

    /*
     * Final tensor distributions.
     */
    os << "\n";
    os << "Final Tensor Distributions\n";
    os << "--------------------------\n";

    for (const auto& n : nodes_) {
        auto it =
            plan_->distributions.find(n.info.id);

        if (it == plan_->distributions.end())
            continue;

        os << "  "
           << tensor_name(n.info.id)
           << " : ";

        print_distribution(it->second);

        os << "\n";
    }

    /*
     * Summary.
     */
    os << "\n";
    os << "Summary\n";
    os << "-------\n";
    os << "  tensors       : "
       << nodes_.size()
       << "\n";

    os << "  operations    : "
       << operations_.size()
       << "\n";

    os << "  devices       : "
       << device_count_
       << "\n";

    os << "  total cost    : "
       << plan_->total_cost
       << "\n";

    os << "====================================================\n";
    os << "\n";
}

// -----------------------------------------------------------------------------
// Reset
// -----------------------------------------------------------------------------

void PlannerEngine::clear()
{
    tensor_ids_.clear();

    nodes_.clear();

    operations_.clear();

    plan_.reset();

    next_tensor_id_ = 0;
    next_operation_id_ = 0;
}