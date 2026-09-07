#include "ggml/Device.hpp"
#include "ggml/MetaDevice.hpp"
#include "ggml/Backend.hpp"
#include "ggml/Context.hpp"
#include "ggml/DeviceAllocator.hpp"
#include "ggml/Scheduler.hpp"
#include "diffusers/pipelines/flux2/Flux2KleinPipeline.hpp"
#include <iostream>
#include <filesystem>
#include <chrono>
#include <thread>


int main() {
    ggml_time_init();
    ggml_log_set([](ggml_log_level, const char* text, void*) { std::cerr << text; }, nullptr);

    ggml_backend_load_all();

    auto gpus = MetaDevice::all(GGML_BACKEND_DEVICE_TYPE_GPU);
    Device cpu(GGML_BACKEND_DEVICE_TYPE_CPU);
    Backend gpus_backend(gpus);
    Backend cpu_backend(cpu);
    Scheduler scheduler({&gpus_backend, &cpu_backend}, 836464);
    Context weights_context(836464);
    DeviceAllocator allocator(weights_context, gpus);

    auto pipeline = std::move(Flux2KleinPipeline::from_pretrained(weights_context, weights_context, weights_context, "../utils/convert-model"));

    allocator.allocate(GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    
    Flux2KleinPipeline::GenerationOptions options;
    options.prompt = "a lovely cat";
    options.width = 256;
    options.height = 256;

    auto images = pipeline(scheduler, weights_context, weights_context, weights_context, gpus, std::move(options));

    //images[0].save("test2.png");
}

#if 0
// ============================================================================
// Tensor-parallel sharding planner -- standalone proof of concept
//
// This file is intentionally self-contained: it mocks the small slice of the
// project's ggml/nn API surface it needs, so it can be built and run
// directly, without CMake:
//
//     g++ -std=c++17 -O2 -o sharding-poc src/main.cpp && ./sharding-poc
//
// The piece that will be migrated into the project is PlannerEngine (plus
// the ggml_backend_meta_split_state types below): it implements the exact
// Engine interface from src/ggml/Engine.hpp and records the graph built
// through it. Everything else (mock ggml types, Context / Tensor / Scope /
// Module framework, main()) is throwaway scaffolding that only exists to
// exercise the planner without pulling in a real backend.
//
// The real usage pattern
// ----------------------
//   1. Run module forward() on a PlannerEngine: the plan phase traces the
//      graph and picks, per tensor, how its data is spread over the devices
//      of the parallel group.
//   2. Allocate the static tensors (weights) with the plan's split states:
//      the plan's callback table is exactly what a
//      ggml_backend_meta_get_split_state_t callback must return for each
//      statically allocated tensor (keyed by tensor name).
//   3. Run the real forward() on the ExecutionEngine: it generates the
//      actual computation nodes. The meta device derives every compute
//      tensor's split state from the callback states and its per-op rules;
//      the planner guarantees that derivation stays in a state the meta
//      backend can execute and that the communication stays minimal.
//
// What the planner does
// ---------------------
// Every tensor in the graph is assigned a distribution -- how its data is
// spread over the devices of the parallel group:
//
//   R      replicated   every device holds the whole tensor
//                          (ggml_backend_meta_split_axis == MIRRORED)
//   S(a)   sharded      device i owns the slice along axis a
//                          (split axis == a; ne[j] is the per-device size)
//   P      partial      device i holds a partial result that must be
//                       AllReduced before it is correct (== PARTIAL)
//
// Axis numbers follow GGML's ne[] order (axis 0 is the fastest dimension)
// and there are at most 4 axes.
//
// Compatibility with the meta backend (ggml/src/ggml-backend-meta.cpp)
// --------------------------------------------------------------------
// The planner only ever plans what the meta backend can execute:
//
//  * The meta device has exactly ONE collective: an AllReduce at the
//    boundary of a PARTIAL subgraph (its only communication primitive).
//    There is no AllGather, no ReduceScatter, no AllToAll. A sharded
//    tensor is consumed sharded through the per-op rules below; a full
//    tensor is (re-)produced only by a row-parallel mul_mat + the
//    implicit AllReduce. Hence the only "bridge" the DP may use is
//        P -> R    AllReduce, cost 0.5 * w_comm * (n-1)/n
//    (the meta reduces with a butterfly, log2(n) steps; the 0.5 factor
//    is a cost-model choice, like all the weights below).
//  * The per-op state rules mirrored from the meta backend:
//      mul_mat (lhs = weight, rhs = activation, out ne = {w->ne[1], a->ne[1..3]}):
//          (R,  {R,  R  }) -> R     (M, M) -> M
//          (S(0), {S(1), R}) -> S(0)  column-parallel: weight->ne[1] sharded
//          (S(1), {R,  S(1)}) -> S(1) token-parallel: activation->ne[1] sharded
//          (P,   {S(0), S(0)}) -> P  row-parallel: contract dim sharded;
//                                     the meta GGML_ASSERTs that the weight
//                                     and activation splits are equal, so
//                                     the planner only offers this candidate
//                                     when weight->ne[0] == activation->ne[0]
//          anything else -> the meta GGML_ABORTs
//      binary add/sub/mul/div (ggml broadcasts the 2nd arg against the 1st):
//          (R, {R, R}) -> R
//          (S(a), {S(a), S(a)}) -> S(a)   both operands sharded equally
//                                         (a < both ranks)
//          (S(a), {S(a), R}) -> S(a)      2nd operand has ne[a] == 1
//                                         (a broadcast, e.g. a bias)
//          the sharded operand must be the 1st (ggml's broadcast order)
//      elementwise unary sqrt/log/sin/cos/sigmoid/scale/clamp/leaky_relu/unary:
//          state carries over: (R, {R}), (S(a), {S(a)})
//      sum_rows and friends (per-row ops):
//          (R, {R}), (S(a), {S(a)}) for a >= 1 -- the axis is preserved
//          (the meta asserts axis != 0 and returns the src state unchanged)
//      norm and rms_norm (the meta's handle_per_row, like sum_rows):
//          (R, {R}), (S(a), {S(a)}) for a >= 1 -- axis 0 is the reduced
//          axis; a src sharded along it GGML_ASSERTs
//      rope_ext (the meta's handle_rope): the position src must be R; the
//          first src's state carries over unchanged, any axis (even 0)
//      get_rows (the meta's handle_get_rows):
//          (R, {R, R}), (S(0), {S(0), R}) -- the data may be sharded along
//          axis 0 with replicated row indices
//      flash_attn_ext (the meta's handle_flash_attn_ext): q, k and v are
//          HARD-asserted to be S(2) (the sequence axis of the rank-4
//          {head_dim, heads, seq, batch} layout), the mask must be R and
//          the output is S(1). There is no replicated candidate: a graph
//          that uses attention must route a shard onto axis 2 (the
//          planner does it through zero-cost reshape/permute views)
//      conv_2d_direct, pool_2d, upscale and interpolate: GGML_OP_CONV_2D,
//          POOL_2D and UPSCALE (ggml_interpolate creates a GGML_OP_UPSCALE
//          node) all go through the meta's handle_generic with
//          scalar_only = true -- a sharded input ABORTS, only R is planned
//      reshape/cont (ggml's handle_reshape, ported verbatim below) and
//      permute (S(b) -> S(i) where the permute puts axis b at position i):
//          exact axis remap, zero cost
//      concat (ggml dim d):
//          (R, {R, R}), (S(a), {S(a), S(a)} | {S(a), R} | {R, S(a)})
//          for a != d
//      repeat: state carries over; the target is a compute leaf (MIRRORED),
//          so only a replicated source works
//      fill: the output takes the shape template's state (a local memset)
//      cpy: like reshape for sharded sources (the meta routes CPY through
//          handle_reshape), zero extra state logic
//      dup: the meta runs it with scalar_only -- a DUP of a sharded tensor
//          ABORTS. Only (R, {R}) is planned. (Consequence for the project:
//          in-place op wrappers that clone first, e.g. Tensor::clamp(),
//          are not sharding-compatible; the demo below clamps in place.)
//      Not handled by the meta backend at all (default case of the switch
//          -> GGML_ABORT "ggml op not implemented"): exp, cast.
//          The planner gives these nodes zero candidates, so any graph
//          using them is reported infeasible with a clear reason.
//  * A P tensor is produced only by a row-parallel mul_mat, and every
//    consumer of it sees MIRRORED (the meta derives source states with
//    assume_sync = true). In the plan: the P node carries the AllReduce
//    bridge to its R consumer. (The meta may *delay* the AllReduce across
//    a trailing linear op like a scalar mul/scale to fuse it into the
//    reduce -- a scheduling detail that does not change any state.)
//  * Graph inputs and compute leaves (tensors the pipeline creates with
//    ggml_set_input, op GGML_OP_NONE) live in the compute buffer: the meta
//    derives them as MIRRORED and never calls the callback for them; the
//    planner fixes them to R.
//  * The plan materializes static tensor splits with llama.cpp-style
//    near-uniform per-device sizes (ne[j] = boundary(j+1) - boundary(j),
//    boundary(i) = ne[axis] * i / n), which keeps the meta's
//    split_states_equal asserts happy for derived compute tensors.
//
// While tracing, every op records its "candidates": the distributions it
// can produce natively, the distribution its inputs must be in for that,
// and the compute cost. finalize() then runs a dynamic program over
// (node, required distribution):
//
//   exact(n, d) = cheapest way for node n to PRODUCE distribution d
//               = min over candidates c with c.output == d of
//                   c.comp_cost + sum over inputs best(input_i, c.input_i)
//   best(n, d)  = cheapest way for node n to SATISFY distribution d, i.e.
//               produce some d' it can emit and bridge d' -> d:
//               = min over producible d' of exact(n, d') + bridge(d', d)
//
// Nodes are traced in topological order (inputs first), so the recursion
// is strictly acyclic and memoization terminates.
//
// Consistency: the meta backend derives exactly one split state per
// tensor, so a tensor (param or compute node) consumed in two different
// distributions cannot be planned; finalize() detects the conflict and
// marks the plan infeasible.
//
// verify() re-derives every node's state from the callback table and the
// per-op rules (topological order, P producers visible as R to their
// consumers) and compares against the plan -- a cheap proof that the plan
// is exactly what the meta device will derive.
//
// Cost model (flat weights; tune to taste)
// -----------------------------------------
//   w_comp  compute of one full (replicated) op
//           a sharded op computes 1/n of the work -> w_comp / n
//   w_comm  cost of moving a unit tensor between devices; the AllReduce
//           above is 0.5 * w_comm * (n-1)/n
//   w_mem   per-device storage of a unit tensor; a replicated param
//           stores n copies -> n * w_mem, a sharded param one -> w_mem
// ============================================================================

#include "nn/Module.hpp"
#include "nn/Parameter.hpp"
#include "nn/CreateEmptyParametersVisitor.hpp"
#include "ggml/PlannerEngine.hpp"

class Linear : public Module {
public:
    Linear(int64_t in_features, int64_t out_features, bool bias = true) {
        // PyTorch convention: weight is [out_features, in_features].
        modules["weight"] = std::make_shared<Parameter>(Tensor::Shape{out_features, in_features});
        if (bias)
            modules["bias"] = std::make_shared<Parameter>(Tensor::Shape{out_features});
    }

    Tensor forward(Scope scope, Tensor x) {
        // Mirrors the project's nn/Linear::forward (src/nn/Linear.cpp).
        auto weight = std::static_pointer_cast<Parameter>(modules["weight"])->forward();

        auto y = scope.engine().mul_mat(*weight, *x);

        if (modules.count("bias")) {
            auto bias = std::static_pointer_cast<Parameter>(modules["bias"])->forward();
            y = scope.engine().add(y, *bias);
        }

        Tensor::Shape shape = x.shape();
        shape[shape.rank() - 1] = weight.shape()[0];

        return Tensor(y, shape);
    }
};

class ReLU : public Module {
public:
    // A sharding-compatible ReLU: the meta backend cannot DUP a sharded
    // tensor (its handle_generic runs with scalar_only), and the project's
    // Tensor::clamp() clones first, so the sharding-friendly path calls
    // the engine's clamp() directly.
    Tensor forward(Scope scope, Tensor x) {
        return Tensor(scope.engine().clamp(*x, 0.0f, std::numeric_limits<float>::infinity()), x.shape());
    }
};

class SiLU : public Module {
public:
    // Mirrors the project's nn/SiLU (src/nn/SiLU.hpp): SiLU(x) = x *
    // sigmoid(x), composed of ops the meta backend supports (MUL +
    // UNARY carry the state over) -- ggml_silu (GGML_OP_SILU) has no
    // split-state rule there.
    Tensor forward(Scope scope, Tensor x) {
        return x * Tensor(scope.engine().sigmoid(*x), x.shape());
    }
};

template <class Act>
class MLP : public Module {
public:
    MLP() {
        modules["fc1"] = std::make_shared<Linear>(8, 16);
        modules["act"] = std::make_shared<Act>();
        modules["fc2"] = std::make_shared<Linear>(16, 8);
    }

    Tensor forward(Scope scope, Tensor x) {
        x = std::static_pointer_cast<Linear>(modules["fc1"])->forward(scope, x);
        x = std::static_pointer_cast<Act>(modules["act"])->forward(scope, x);
        return std::static_pointer_cast<Linear>(modules["fc2"])->forward(scope, x);
    }
};

// Demo 3 module: an attention block built on flash_attn_ext.
//
// The meta's flash-attn rule requires q, k, v in S(2) (the sequence axis of
// the rank-4 {head_dim, heads, seq, batch} layout) and has no replicated
// candidate. With a replicated block input, the planner's only feasible
// route is a column-parallel projection (shard on the feature axis) moved
// onto the sequence axis by zero-cost views:
//
//   x {hidden, seq, batch} R
//   q2 = mul_mat(w_q, x)               w_q S(1) -> q2 S(0)  (column-parallel)
//   q4 = reshape_4d(q2, hd, H, S, B)                  S(0) -> S(1)
//   q' = permute(q4, 0, 2, 1, 3)                      S(1) -> S(2)
//   attn = flash_attn_ext(q', k', v', mask)           S(2)^3, R -> S(1)
//   attn2 = reshape_3d(attn, hidden, S, B)            S(1) -> S(0)
//   out = mul_mat(w_o, attn2)                         (S(0), S(0)) -> P --AllReduce--> R
class AttentionBlock : public Module {
public:
    AttentionBlock(int64_t hidden, int64_t heads)
        : hidden_(hidden), heads_(heads), head_dim_(hidden / heads) {
        modules["w_q"] = std::make_shared<Parameter>(Tensor::Shape{hidden, hidden});
        modules["w_k"] = std::make_shared<Parameter>(Tensor::Shape{hidden, hidden});
        modules["w_v"] = std::make_shared<Parameter>(Tensor::Shape{hidden, hidden});
        modules["w_o"] = std::make_shared<Parameter>(Tensor::Shape{hidden, hidden});
    }

    Tensor forward(Scope scope, Tensor x) {
        auto& e = scope.engine();
        const int64_t seq = x.shape()[1];
        const int64_t batch = x.shape()[2];
        const int64_t head_dim = head_dim_;

        auto w_q = std::static_pointer_cast<Parameter>(modules["w_q"])->forward();
        auto w_k = std::static_pointer_cast<Parameter>(modules["w_k"])->forward();
        auto w_v = std::static_pointer_cast<Parameter>(modules["w_v"])->forward();
        auto w_o = std::static_pointer_cast<Parameter>(modules["w_o"])->forward();

        auto q2 = e.mul_mat(*w_q, *x);   // ggml_tensor*, {hidden, seq, batch}
        auto k2 = e.mul_mat(*w_k, *x);
        auto v2 = e.mul_mat(*w_v, *x);

        // rank 4: GGML {head_dim, heads, seq, batch} == PyTorch (batch, seq, heads, head_dim)
        auto q4 = e.reshape_4d(q2, head_dim, heads_, seq, batch);
        auto k4 = e.reshape_4d(k2, head_dim, heads_, seq, batch);
        auto v4 = e.reshape_4d(v2, head_dim, heads_, seq, batch);

        // GGML {head_dim, seq, heads, batch}: the shard (heads, S(1)) moves to axis 2.
        auto qp = e.permute(q4, 0, 2, 1, 3);
        auto kp = e.permute(k4, 0, 2, 1, 3);
        auto vp = e.permute(v4, 0, 2, 1, 3);

        // Attention mask: a compute leaf (fixed R), GGML {1, 1, 1, batch}.
        auto mask = Tensor::empty<float>(Tensor::Shape{1, 1, 1, batch}).input();

        // ggml: out ne = {v->ne[0], q->ne[2], q->ne[1], q->ne[3]} = {head_dim, heads, seq, batch}
        auto attn = e.flash_attn_ext(qp, kp, vp, *mask, 0.5f, 0.0f, 0.0f);

        // Back to {hidden, seq, batch} for the output projection.
        auto attn2 = e.reshape_3d(attn, hidden_, seq, batch);   // (S(1) -> S(0))

        auto out = e.mul_mat(*w_o, attn2);   // (S(0), S(0)) -> P
        return Tensor(out, Tensor::Shape{batch, seq, hidden_});
    }

private:
    int64_t hidden_, heads_, head_dim_;
};

// Demo 4 module: a projection path over rope_ext and get_rows.
//
// handle_rope asserts only that the position src is MIRRORED and carries a's
// state over (any axis, even 0); handle_get_rows lets the data be sharded
// along axis 0 with replicated indices. The shard is created by a
// column-parallel mul_mat and consumed by a row-parallel projection
// (the only P -> R bridge):
//
//   x {hidden, seq, batch} R
//   y = mul_mat(w1, x)                  w1 S(1) -> y S(0)   (column-parallel)
//   r = rope_ext(y, pos, nullptr)                   S(0) -> S(0)
//   g = get_rows(r, idx)                         (S(0), R) -> S(0)
//   out = mul_mat(w2, g)                        (S(0), S(0)) -> P --AllReduce--> R
class RopeGetRows : public Module {
public:
    RopeGetRows(int64_t hidden, int64_t n_rows)
        : hidden_(hidden), n_rows_(n_rows) {
        modules["w1"] = std::make_shared<Parameter>(Tensor::Shape{hidden, hidden});
        modules["w2"] = std::make_shared<Parameter>(Tensor::Shape{n_rows_, hidden});
    }

    Tensor forward(Scope scope, Tensor x) {
        auto& e = scope.engine();
        auto w1 = std::static_pointer_cast<Parameter>(modules["w1"])->forward();
        auto w2 = std::static_pointer_cast<Parameter>(modules["w2"])->forward();

        auto y = e.mul_mat(*w1, *x);   // ggml_tensor*, {hidden, seq, batch}

        // Positions: a compute leaf (fixed R), GGML {1, seq, 1}.
        auto pos = Tensor::empty<float>(Tensor::Shape{1, x.shape()[1], 1}).input();
        auto r = e.rope_ext(y, *pos, nullptr, 0, 0, 0, 10000.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f);

        // Row indices: ggml_get_rows requires b = {n_rows, a->ne[2], a->ne[3], 1};
        // for a = {hidden, seq, batch} that is {n_rows, batch, 1, 1}.
        auto idx = Tensor::empty<float>(Tensor::Shape{x.shape()[0], n_rows_}).input();
        auto g = e.get_rows(r, *idx);   // {hidden, n_rows, batch, 1}

        auto out = e.mul_mat(*w2, g);   // {n_rows, n_rows, batch, 1}
        return Tensor(out, Tensor::Shape{x.shape()[0], n_rows_, n_rows_});
    }

private:
    int64_t hidden_, n_rows_;
};

// Demo 5 module: a VAE-style upsampler built on the vision ops.
//
// conv_2d_direct, pool_2d, upscale and interpolate are scalar_only in the
// meta backend (a sharded input aborts), so the whole block is forced to
// the replicated form -- including the norm nodes riding along.
class Upsampler : public Module {
public:
    Upsampler(int64_t in_channels, int64_t out_channels)
        : out_channels_(out_channels) {
        modules["w_conv"] = std::make_shared<Parameter>(Tensor::Shape{3, 3, in_channels, out_channels});
    }

    Tensor forward(Scope scope, Tensor x) {
        // x: PyTorch (batch, channels, height, width) == GGML {w, h, c, n}
        auto& e = scope.engine();
        auto w = std::static_pointer_cast<Parameter>(modules["w_conv"])->forward();
        const int64_t batch = x.shape()[0], h = x.shape()[2], wdt = x.shape()[3];

        // 3x3, pad 1: same spatial size.
        auto c = e.conv_2d_direct(*w, *x, 1, 1, 1, 1, 1, 1);
        auto n = e.norm(c, 1e-6f);
        auto u = e.upscale(n, 2, GGML_SCALE_MODE_NEAREST);
        auto i = e.interpolate(u, 4 * wdt, 4 * h, out_channels_, batch, GGML_SCALE_MODE_BILINEAR);
        auto p = e.pool_2d(i, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0.0f, 0.0f);
        return Tensor(p, Tensor::Shape{batch, out_channels_, 2 * h, 2 * wdt});
    }

private:
    int64_t out_channels_;
};

// ============================================================================
// Demo
// ============================================================================

void print_callback_table(const PlannerEngine::Plan& plan) {
    std::cout << "meta device callback table (ggml_backend_meta_split_state per static tensor):\n";
    for (const auto& [name, st] : plan.callback_states) {
        std::cout << "  " << std::left << std::setw(18) << name << std::right;
        switch (st.axis) {
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED: std::cout << "  MIRRORED      ne=[]"; break;
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL:  std::cout << "  PARTIAL       ne=[]"; break;
            default:
                std::cout << "  axis " << (int)st.axis << "    ne=[";
                for (size_t j = 0; j < plan.device_count; ++j)
                    std::cout << (j ? ", " : "") << st.ne[j];
                std::cout << "]";
        }
        std::cout << "\n";
    }
}

int main() {
    ggml_time_init();
    ggml_backend_load_all();

    {
        // Demo 1: an MLP that the meta backend can plan. The activation is
        // a ReLU (in-place clamp): the meta backend cannot DUP a sharded
        // tensor, so the sharding-friendly path skips Tensor::clamp's clone.
        PlannerEngine planner(/*device_count=*/2, /*w_comm=*/0.5, /*w_comp=*/1.0, /*w_mem=*/0.1);
        Scope scope(planner);

        // Graph input: rank-4 activation, PyTorch shape (2, 3, 4, 8)
        // == GGML ne {8, 4, 3, 2}; created like a pipeline input (a
        // compute leaf: fixed R).
        Tensor x = Tensor::empty<float>(Tensor::Shape{2, 3, 4, 8}).input();

        MLP<ReLU> model;
        CreateEmptyParametersVisitor visitor;
        model.accept(visitor);

        (void)model.forward(scope, x);

        std::cout << "traced graph:\n" << planner.dump_trace() << "\n";
        auto plan = planner.finalize();
        std::cout << plan.to_string();

        std::string error;
        if (planner.verify(plan, error))
            std::cout << "verification: OK -- the plan matches the meta backend's split-state derivation\n";
        else
            std::cout << "verification FAILED: " << error << "\n";

        print_callback_table(plan);
    }

    {
        // Demo 2: the same MLP with a SiLU activation, written as
        // x * sigmoid(x) (the project's nn/SiLU does the same): both ops
        // carry the split state over, so the block plans exactly like the
        // clamp version -- and stays sharding-compatible.
        PlannerEngine planner(/*device_count=*/2, /*w_comm=*/0.5, /*w_comp=*/1.0, /*w_mem=*/0.1);
        Scope scope(planner);
        Tensor x = Tensor::empty<float>(Tensor::Shape{2, 3, 4, 8}).input();

        MLP<SiLU> model;
        CreateEmptyParametersVisitor visitor;
        model.accept(visitor);

        (void)model.forward(scope, x);

        std::cout << "\n";
        auto plan = planner.finalize();
        std::cout << plan.to_string();

        std::string error;
        if (planner.verify(plan, error))
            std::cout << "verification: OK -- the plan matches the meta backend's split-state derivation\n";
        else
            std::cout << "verification FAILED: " << error << "\n";

        print_callback_table(plan);
    }

    {
        // Demo 3: an attention block on flash_attn_ext. The meta backend
        // hard-asserts q, k, v in S(2) and offers no replicated candidate,
        // so with a replicated block input the only feasible plan routes a
        // column-parallel shard onto the sequence axis through zero-cost
        // reshape/permute views (see the module). The block ends with a
        // row-parallel projection: P -> AllReduce -> R.
        PlannerEngine planner(/*device_count=*/2, /*w_comm=*/0.5, /*w_comp=*/1.0, /*w_mem=*/0.1);
        Scope scope(planner);

        // Block input: PyTorch (batch, seq, hidden) == GGML {hidden, seq, batch}.
        Tensor x = Tensor::empty<float>(Tensor::Shape{2, 4, 8}).input();

        AttentionBlock model(/*hidden=*/8, /*heads=*/4);
        CreateEmptyParametersVisitor visitor;
        model.accept(visitor);

        (void)model.forward(scope, x);

        std::cout << "\nDemo 3: attention block (flash_attn_ext)\n";
        std::cout << "traced graph:\n" << planner.dump_trace() << "\n";
        auto plan = planner.finalize();
        std::cout << plan.to_string();

        std::string error;
        if (planner.verify(plan, error))
            std::cout << "verification: OK -- the plan matches the meta backend's split-state derivation\n";
        else
            std::cout << "verification FAILED: " << error << "\n";

        print_callback_table(plan);
    }

    {
        // Demo 4: rope_ext + get_rows. The shard is created by a
        // column-parallel mul_mat (feature axis S(0)); rope carries the
        // state over (any axis) and get_rows keeps the S(0) shard with
        // replicated indices; the row-parallel projection ends the block
        // (P -> AllReduce -> R).
        PlannerEngine planner(/*device_count=*/2, /*w_comm=*/0.5, /*w_comp=*/1.0, /*w_mem=*/0.1);
        Scope scope(planner);
        Tensor x = Tensor::empty<float>(Tensor::Shape{2, 4, 8}).input();

        RopeGetRows model(/*hidden=*/8, /*n_rows=*/6);
        CreateEmptyParametersVisitor visitor;
        model.accept(visitor);

        (void)model.forward(scope, x);

        std::cout << "\nDemo 4: rope_ext + get_rows\n";
        auto plan = planner.finalize();
        std::cout << plan.to_string();

        std::string error;
        if (planner.verify(plan, error))
            std::cout << "verification: OK -- the plan matches the meta backend's split-state derivation\n";
        else
            std::cout << "verification FAILED: " << error << "\n";

        print_callback_table(plan);
    }

    {
        // Demo 5: a VAE-style upsampler: conv_2d_direct, norm, upscale,
        // interpolate, pool_2d. Every one of these ops is scalar_only in
        // the meta backend, so the whole block must be replicated -- any
        // shard would abort, and the all-replicated plan is the correct
        // (and only) outcome.
        PlannerEngine planner(/*device_count=*/2, /*w_comm=*/0.5, /*w_comp=*/1.0, /*w_mem=*/0.1);
        Scope scope(planner);

        // Block input: PyTorch (batch, channels, h, w) == GGML {w, h, c, n}.
        Tensor x = Tensor::empty<float>(Tensor::Shape{2, 4, 8, 8}).input();

        Upsampler model(/*in_channels=*/4, /*out_channels=*/8);
        CreateEmptyParametersVisitor visitor;
        model.accept(visitor);

        (void)model.forward(scope, x);

        std::cout << "\nDemo 5: VAE-style upsampler (conv_2d_direct, upscale, interpolate, pool_2d)\n";
        auto plan = planner.finalize();
        std::cout << plan.to_string();

        std::string error;
        if (planner.verify(plan, error))
            std::cout << "verification: OK -- the plan matches the meta backend's split-state derivation\n";
        else
            std::cout << "verification FAILED: " << error << "\n";

        print_callback_table(plan);
    }

    return 0;
}
#endif
