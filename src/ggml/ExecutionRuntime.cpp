#include "ggml/ExecutionRuntime.hpp"
#include "ggml/Scope.hpp"
#include "ggml/Context.hpp"
#include "ggml/Allocator.hpp"
#include "ggml/Scheduler.hpp"
#include "ggml/Computation.hpp"
#include <iostream>

ExecutionRuntime ExecutionRuntime::Default;

static std::string format_bytes(std::size_t bytes) {
    if (bytes >= 1024 * 1024) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "%.2f MB", double(bytes) / (1024.0 * 1024.0));
        return buf;
    }
    if (bytes >= 1024)
        return std::to_string(bytes / 1024) + " KB";
    return std::to_string(bytes) + " B";
}

static std::size_t count_tensors(Context& ctx) {
    std::size_t n = 0;
    for (ggml_tensor* t = ggml_get_first_tensor(*ctx); t != nullptr; t = ggml_get_next_tensor(*ctx, t))
        ++n;
    return n;
}

void ExecutionRuntime::run(Scheduler& scheduler, Allocator& weights_allocator, Allocator& state_allocator, std::mt19937& rng, ComputationDescription& desc) const {
    // The one-shot allocation point: all the module forwards have been
    // called — all the low-level ggml tensor chains exist. Allocate the
    // long-lived contexts.

    if (desc.weights) {
        weights_allocator.allocate(*desc.weights);
        /*std::cerr << "  [weights] buffer: "
                    << format_bytes(ggml_backend_buffer_get_size(desc.weights->buffer()))
                    << "  (one-shot, pinned — as today's USAGE_WEIGHTS)\n";*/
    }

    if (desc.state) {
        state_allocator.allocate(*desc.state);
        /*std::cerr << "  [state]   buffer: "
                    << format_bytes(ggml_backend_buffer_get_size(desc.state->buffer()))
                    << "  (one-shot, lives for the whole generation)\n";*/
    }

    // The weights: the one-shot bindings are streamed once into the
    // weights buffer.
    if (desc.weights) {
        std::cerr << "weights: " << count_tensors(*desc.weights) << " tensors" << std::endl;
        bind(rng, *desc.weights, /*once_only=*/true);
    }

    // The state's one-shot bindings.
    if (desc.state) {
        std::cerr << "state: " << count_tensors(*desc.state) << " tensors" << std::endl;
        bind(rng, desc.state.value(), /*once_only=*/true);
    }

    std::cerr << std::endl;

    for (auto i = 0; i < desc.scopes.size(); ++i) {
        auto& r = desc.scopes[i];

        // The cgraph is built in the temporary context (its memory comes from
        // the context's pool): the body's tensor chain was already recorded
        // there at build time; ggml_build_forward_expand expands it into the
        // graph (the nodes, and the leaves — the bound inputs of this context
        // and the pre-allocated weights / state tensors).
        auto gf = ggml_new_graph_custom(**r.context, r.context->capacity(), /*grads=*/false);

        // Build graph for every output Tensor.
        for (const auto& out : r.outputs)
            ggml_build_forward_expand(gf, *out);

        // Reserve the compute buffer for this graph (it persists on the
        // scheduler and is reused — and grown — for later graphs; in the
        // project this is done once, with a max-size measure graph). In this
        // ggml version reserve_size measures without allocating, so it cannot
        // be the only reserve: alloc_graph would then skip the allocation.
        if (!ggml_backend_sched_reserve(*scheduler, gf))
            throw std::runtime_error("ggml_backend_sched_reserve failed");

        std::cerr << "scope #" << i <<  " graph: " << ggml_graph_n_nodes(gf) << " nodes" << std::endl;

        // Allocate: the galloc plans the graph's temporary tensors with
        // liveness (the compute buffer is sized to the high-water mark of
        // the live set); the weights / state tensors are already allocated
        // and skipped. build_graph reserved the buffer; alloc_graph assigns
        // the tensors' addresses in it.
        if (!ggml_backend_sched_alloc_graph(*scheduler, gf))
            throw std::runtime_error("ggml_backend_sched_alloc_graph failed");

        // The bound inputs (once): written into the freshly allocated graph
        // tensors (ggml_backend_tensor_set).
        bind(rng, *r.context, /*once_only=*/true);

        // Handle repeating computation with re-binding and state carrying.
        if (r.loop) {
            // The graph is a single allocation with multiple computations.
            for (r.iter = 0; r.iter < r.count; ++r.iter) {
                // The re-bindable inputs: rewritten on every (re-)execution —
                // their providers run through the loop's iter clock.
                bind(rng, *r.context, /*once_only=*/false);

                if (ggml_backend_sched_graph_compute(*scheduler, gf) != GGML_STATUS_SUCCESS)
                    throw std::runtime_error("ggml_backend_sched_graph_compute failed");

                // The feedback: the next state is written back into the
                // state cell (on every iteration, including the last — the
                // cell must hold the final state.
                for (auto& [src, dst] : r.feedback)
                    copy(src, dst);
            }
        }
        
        // Otherwise, handle singular computation.
        else {
            // The re-bindable inputs of a single-execution scope have no
            // iter clock to read, so they are written once, right before the
            // (only) execution.
            bind(rng, *r.context, /*once_only=*/false);

            if (ggml_backend_sched_graph_compute(*scheduler, gf) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("ggml_backend_sched_graph_compute failed");
        }

        // Copy output values into state variables.
        for (auto& [src, dst] : r.saves)
            copy(src, dst);
    }
}

void ExecutionRuntime::bind(std::mt19937& rng, Context& context, bool once_only) const {
    for (auto& [tensor, binding] : context.bindings()) {
        if (binding.unbound || binding.once != once_only)
            continue;

        auto bytes = binding.provider(rng);

        if (bytes.size() != ggml_nbytes(*tensor))
            throw std::runtime_error(std::string("bind(): size mismatch for '") + ggml_get_name(*tensor) + "'");

        ggml_backend_tensor_set(*tensor, bytes.data(), 0, bytes.size());
    }
}

void ExecutionRuntime::copy(const Tensor& src, const Tensor& dst) const {
    // A tensor in a temporary context is copied into the state context
    // (ggml_backend_tensor_copy — the project's Context::copy).
    ggml_backend_tensor_copy(*src, *dst);
}


// -------------------------------------------------------------------------
// Tensor creation / initialization
// -------------------------------------------------------------------------

ggml_tensor* ExecutionRuntime::new_tensor(
    ggml_type type,
    int n_dims,
    const int64_t* ne
) {
    return ggml_new_tensor(*Scope::context(), type, n_dims, ne);
}

ggml_tensor* ExecutionRuntime::new_tensor_1d(
    ggml_type type,
    int64_t ne0
) {
    return ggml_new_tensor_1d(*Scope::context(), type, ne0);
}

void ExecutionRuntime::set_input(
    ggml_tensor* tensor
) {
    ggml_set_input(tensor);
}

void ExecutionRuntime::set_param(
    ggml_tensor* tensor
) {
    // There is no GGML counterpart for this.
}

void ExecutionRuntime::set_output(
    ggml_tensor* tensor
) {
    ggml_set_output(tensor);
}

ggml_tensor* ExecutionRuntime::fill(
    ggml_tensor* tensor,
    float value
) {
    return ggml_fill(*Scope::context(), tensor, value);
}

// -----------------------------------------------------------------------------
// Copy / cast
// -----------------------------------------------------------------------------

ggml_tensor* ExecutionRuntime::cont(
    ggml_tensor* tensor
) {
    return ggml_cont(*Scope::context(), tensor);
}

ggml_tensor* ExecutionRuntime::dup(
    ggml_tensor* tensor
) {
    return ggml_dup(*Scope::context(), tensor);
}

ggml_tensor* ExecutionRuntime::cast(
    ggml_tensor* tensor,
    ggml_type type
) {
    return ggml_cast(*Scope::context(), tensor, type);
}

ggml_tensor* ExecutionRuntime::cpy(
    ggml_tensor* src,
    ggml_tensor* dst
) {
    return ggml_cpy(*Scope::context(), src, dst);
}

// -----------------------------------------------------------------------------
// Unary arithmetic
// -----------------------------------------------------------------------------

ggml_tensor* ExecutionRuntime::sqrt(
    ggml_tensor* tensor
) {
    return ggml_sqrt(*Scope::context(), tensor);
}

ggml_tensor* ExecutionRuntime::exp(
    ggml_tensor* tensor
) {
    // GGML_OP_EXP has no split-state rule in the meta backend; the
    // elementwise GGML_UNARY(EXP) does (handle_generic carries the src
    // state over). Same computation, meta-planable.
    return ggml_unary(*Scope::context(), tensor, GGML_UNARY_OP_EXP);
}

ggml_tensor* ExecutionRuntime::log(
    ggml_tensor* tensor
) {
    return ggml_log(*Scope::context(), tensor);
}

ggml_tensor* ExecutionRuntime::sin(
    ggml_tensor* tensor
) {
    return ggml_sin(*Scope::context(), tensor);
}

ggml_tensor* ExecutionRuntime::cos(
    ggml_tensor* tensor
) {
    return ggml_cos(*Scope::context(), tensor);
}

ggml_tensor* ExecutionRuntime::sigmoid(
    ggml_tensor* tensor
) {
    return ggml_sigmoid(*Scope::context(), tensor);
}

// -----------------------------------------------------------------------------
// Binary arithmetic
// -----------------------------------------------------------------------------

ggml_tensor* ExecutionRuntime::add(
    ggml_tensor* lhs,
    ggml_tensor* rhs
) {
    return ggml_add(*Scope::context(), lhs, rhs);
}

ggml_tensor* ExecutionRuntime::sub(
    ggml_tensor* lhs,
    ggml_tensor* rhs
) {
    return ggml_sub(*Scope::context(), lhs, rhs);
}

ggml_tensor* ExecutionRuntime::mul(
    ggml_tensor* lhs,
    ggml_tensor* rhs
) {
    return ggml_mul(*Scope::context(), lhs, rhs);
}

ggml_tensor* ExecutionRuntime::div(
    ggml_tensor* lhs,
    ggml_tensor* rhs
) {
    return ggml_div(*Scope::context(), lhs, rhs);
}

// -----------------------------------------------------------------------------
// Scalar operations
// -----------------------------------------------------------------------------

ggml_tensor* ExecutionRuntime::scale(
    ggml_tensor* tensor,
    float value
) {
    return ggml_scale(*Scope::context(), tensor, value);
}

ggml_tensor* ExecutionRuntime::clamp(
    ggml_tensor* tensor,
    float min,
    float max
) {
    return ggml_clamp(*Scope::context(), tensor, min, max);
}

// -----------------------------------------------------------------------------
// Matrix operations
// -----------------------------------------------------------------------------

ggml_tensor* ExecutionRuntime::mul_mat(
    ggml_tensor* lhs,
    ggml_tensor* rhs
) {
    return ggml_mul_mat(*Scope::context(), lhs, rhs);
}

// -----------------------------------------------------------------------------
// Reshape
// -----------------------------------------------------------------------------

ggml_tensor* ExecutionRuntime::reshape_1d(
    ggml_tensor* tensor,
    int64_t ne0
) {
    return ggml_reshape_1d(*Scope::context(), tensor, ne0);
}

ggml_tensor* ExecutionRuntime::reshape_2d(
    ggml_tensor* tensor,
    int64_t ne0,
    int64_t ne1
) {
    return ggml_reshape_2d(*Scope::context(), tensor, ne0, ne1);
}

ggml_tensor* ExecutionRuntime::reshape_3d(
    ggml_tensor* tensor,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2
) {
    return ggml_reshape_3d(*Scope::context(), tensor, ne0, ne1, ne2);
}

ggml_tensor* ExecutionRuntime::reshape_4d(
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

ggml_tensor* ExecutionRuntime::permute(
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

ggml_tensor* ExecutionRuntime::view_1d(
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

ggml_tensor* ExecutionRuntime::view_2d(
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

ggml_tensor* ExecutionRuntime::view_3d(
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

ggml_tensor* ExecutionRuntime::view_4d(
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

ggml_tensor* ExecutionRuntime::repeat(
    ggml_tensor* tensor,
    ggml_tensor* target
) {
    return ggml_repeat(*Scope::context(), tensor, target);
}

// -----------------------------------------------------------------------------
// Concatenation
// -----------------------------------------------------------------------------

ggml_tensor* ExecutionRuntime::concat(
    ggml_tensor* a,
    ggml_tensor* b,
    int dim
) {
    return ggml_concat(*Scope::context(), a, b, dim);
}

// -----------------------------------------------------------------------------
// Reduction
// -----------------------------------------------------------------------------

ggml_tensor* ExecutionRuntime::sum_rows(
    ggml_tensor* tensor
) {
    return ggml_sum_rows(*Scope::context(), tensor);
}

ggml_tensor * ExecutionRuntime::flash_attn_ext(
    ggml_tensor* q,
    ggml_tensor* k,
    ggml_tensor* v,
    ggml_tensor* mask,
    float scale,
    float max_bias,
    float logit_softcap)
{
    return ggml_flash_attn_ext(*Scope::context(), q, k, v, mask, scale, max_bias, logit_softcap);
}

ggml_tensor * ExecutionRuntime::conv_2d_direct(
    ggml_tensor* a,
    ggml_tensor* b,
    int s0,
    int s1,
    int p0,
    int p1,
    int d0,
    int d1)
{
    return ggml_conv_2d_direct(*Scope::context(), a, b, s0, s1, p0, p1, d0, d1);
}

ggml_tensor* ExecutionRuntime::get_rows(
    ggml_tensor* a,
    ggml_tensor* b)
{
    return ggml_get_rows(*Scope::context(), a, b);
}

ggml_tensor* ExecutionRuntime::norm(
    ggml_tensor* a,
    float eps) 
{
    return ggml_norm(*Scope::context(), a, eps);
}

ggml_tensor* ExecutionRuntime::rms_norm(
    ggml_tensor* a,
    float eps)
{
    return ggml_rms_norm(*Scope::context(), a, eps);
}

ggml_tensor* ExecutionRuntime::rope_ext(
    ggml_tensor* a,
    ggml_tensor* b,
    ggml_tensor* c,
    int n_dims,
    int mode,
    int n_ctx_orig,
    float freq_base,
    float freq_scale,
    float ext_factor,
    float attn_factor,
    float beta_fast,
    float beta_slow) 
{
    return ggml_rope_ext(*Scope::context(), a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
}

ggml_tensor* ExecutionRuntime::pool_2d(
    ggml_tensor* a,
    ggml_op_pool op,
    int k0,
    int k1,
    int s0,
    int s1,
    float p0,
    float p1) 
{
    return ggml_pool_2d(*Scope::context(), a, op, k0, k1, s0, s1, p0, p1);
}

ggml_tensor* ExecutionRuntime::interpolate(
    ggml_tensor* a,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2,
    int64_t ne3,
    uint32_t mode) 
{
    return ggml_interpolate(*Scope::context(), a, ne0, ne1, ne2, ne3, mode);
}

ggml_tensor* ExecutionRuntime::upscale(
    ggml_tensor* a,
    int scale_factor,
    ggml_scale_mode mode)
{
    return ggml_upscale(*Scope::context(), a, scale_factor, mode);
}
