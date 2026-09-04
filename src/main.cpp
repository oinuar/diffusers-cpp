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
//      elementwise unary sqrt/log/sin/cos/scale/clamp/leaky_relu/unary:
//          state carries over: (R, {R}), (S(a), {S(a)})
//      sum_rows and friends (per-row ops):
//          (R, {R}), (S(a), {S(a)}) for a >= 1 -- the axis is preserved
//          (the meta asserts axis != 0 and returns the src state unchanged)
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
//          -> GGML_ABORT "ggml op not implemented"): neg, abs, exp, cast,
//          silu. The planner gives these nodes zero candidates, so any
//          graph using them is reported infeasible with a clear reason.
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

#include <algorithm>
#include <array>
#include <cassert>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// ============================================================================
// Mock GGML core (standalone build only; the real headers are not needed)
// ============================================================================

typedef int ggml_type;

struct ggml_tensor {
    char name[64] = {0};   // set by the (mock) weight loader, read by the planner
    ggml_type type = 0;    // needed for ggml_blck_size (axis-0 split alignment)
};
struct ggml_context { int dummy; };

constexpr ggml_type kMockType = 0;   // the mock graph is dtype-uniform (f32-like, blck size 1)

int ggml_blck_size(ggml_type) { return 1; }   // mock dtypes are f32-like

void ggml_time_init() {}
void ggml_backend_load_all() {}

void ggml_set_name(ggml_tensor* t, const char* name) {
    std::strncpy(t->name, name, sizeof(t->name) - 1);
    t->name[sizeof(t->name) - 1] = 0;
}

// ============================================================================
// GGML meta split types
//
// Copied verbatim from ggml/include/ggml-backend.h -- the plan's callback
// table must return exactly these structures, so they live here instead of
// in the "throwaway" mock section. Keep them in sync with the header.
// ============================================================================

#define GGML_BACKEND_META_MAX_DEVICES 16

enum ggml_backend_meta_split_axis {
    // tensor split by tensor dimensions:
    GGML_BACKEND_SPLIT_AXIS_0 = 0,
    GGML_BACKEND_SPLIT_AXIS_1 = 1,
    GGML_BACKEND_SPLIT_AXIS_2 = 2,
    GGML_BACKEND_SPLIT_AXIS_3 = 3,

    GGML_BACKEND_SPLIT_AXIS_MIRRORED = 10, // all values on all backends
    GGML_BACKEND_SPLIT_AXIS_PARTIAL  = 11, // each backend has a partial sum

    // for internal bookkeeping only:
    GGML_BACKEND_SPLIT_AXIS_NONE    = 98,
    GGML_BACKEND_SPLIT_AXIS_UNKNOWN = 99,
};

struct ggml_backend_meta_split_state {
    enum ggml_backend_meta_split_axis axis;

    // for tensors with axis >= 0 && axis < GGML_MAX_DIMS:
    //   - each device has a slice of the tensor along the split axis
    //   - most tensors have n_segments == 1 and a contiguous slice of the tensor data
    //   - some tensors have an inhomogenenous data layout along the split axis,
    //     those tensors are divided into segments which are each individually split across devices
    //   - ne has one entry per segment and device and that segment repeats nr times,
    //     in total when accounting for repetitions the segments add up to ggml_tensor::ne for that axis,
    //     the outer/inner loops are over segments/devices like [seg0_dev0_r0, seg0_dev1_r0, seg0_dev0_r1, seg0_dev1_r1, seg1_dev0_r0, seg1_dev1_r0],
    //   - for example, a transformer may have a fused QKV matrix rather than 3 matrices, those would be 3 separate segments
    //     that each need to be split individually across devices so that each device gets a slice of Q, K, and V,
    //   - the Q matrix can be larger than the K or V matrices so this can either be expressed as 3 segments or as 2 segments
    //     where the segment for K/V repeats twice
    int64_t  ne[16*GGML_BACKEND_META_MAX_DEVICES];
    uint32_t nr[16];
    uint32_t n_segments;
};

// function to assign split states for statically allocated tensors, compute tensor split states will be assigned to be compatible:
typedef struct ggml_backend_meta_split_state(*ggml_backend_meta_get_split_state_t)(const struct ggml_tensor * tensor, void * userdata);

// ============================================================================
// Engine interface (verbatim copy of src/ggml/Engine.hpp)
// ============================================================================

class Engine {
public:
    virtual ~Engine() = default;

    // Tensor creation / initialization
    virtual ggml_tensor* new_tensor(ggml_context* ctx, ggml_type type, int n_dims, const int64_t* ne) = 0;
    virtual ggml_tensor* new_tensor_1d(ggml_context* ctx, ggml_type type, int64_t ne0) = 0;
    virtual void set_input(ggml_tensor* tensor) = 0;
    virtual ggml_tensor* fill(ggml_tensor* tensor, float value) = 0;

    // Copy / cast
    virtual ggml_tensor* cont(ggml_tensor* tensor) = 0;
    virtual ggml_tensor* dup(ggml_tensor* tensor) = 0;
    virtual ggml_tensor* cast(ggml_tensor* tensor, ggml_type type) = 0;
    virtual ggml_tensor* cpy(ggml_tensor* src, ggml_tensor* dst) = 0;

    // Unary arithmetic
    virtual ggml_tensor* neg(ggml_tensor* tensor) = 0;
    virtual ggml_tensor* abs(ggml_tensor* tensor) = 0;
    virtual ggml_tensor* sqrt(ggml_tensor* tensor) = 0;
    virtual ggml_tensor* exp(ggml_tensor* tensor) = 0;
    virtual ggml_tensor* log(ggml_tensor* tensor) = 0;
    virtual ggml_tensor* sin(ggml_tensor* tensor) = 0;
    virtual ggml_tensor* cos(ggml_tensor* tensor) = 0;

    // Binary arithmetic
    virtual ggml_tensor* add(ggml_tensor* lhs, ggml_tensor* rhs) = 0;
    virtual ggml_tensor* sub(ggml_tensor* lhs, ggml_tensor* rhs) = 0;
    virtual ggml_tensor* mul(ggml_tensor* lhs, ggml_tensor* rhs) = 0;
    virtual ggml_tensor* div(ggml_tensor* lhs, ggml_tensor* rhs) = 0;

    // Scalar arithmetic
    virtual ggml_tensor* scale(ggml_tensor* tensor, float value) = 0;
    virtual ggml_tensor* clamp(ggml_tensor* tensor, float min, float max) = 0;

    // Matrix operations
    virtual ggml_tensor* mul_mat(ggml_tensor* lhs, ggml_tensor* rhs) = 0;

    // Reshape
    virtual ggml_tensor* reshape_1d(ggml_tensor* tensor, int64_t ne0) = 0;
    virtual ggml_tensor* reshape_2d(ggml_tensor* tensor, int64_t ne0, int64_t ne1) = 0;
    virtual ggml_tensor* reshape_3d(ggml_tensor* tensor, int64_t ne0, int64_t ne1, int64_t ne2) = 0;
    virtual ggml_tensor* reshape_4d(ggml_tensor* tensor, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) = 0;

    // Permute / transpose
    virtual ggml_tensor* permute(ggml_tensor* tensor, int axis0, int axis1, int axis2, int axis3) = 0;

    // Views
    virtual ggml_tensor* view_1d(ggml_tensor* tensor, int64_t ne0, size_t offset) = 0;
    virtual ggml_tensor* view_2d(ggml_tensor* tensor, int64_t ne0, int64_t ne1, size_t nb1, size_t offset) = 0;
    virtual ggml_tensor* view_3d(ggml_tensor* tensor, int64_t ne0, int64_t ne1, int64_t ne2, size_t nb1, size_t nb2, size_t offset) = 0;
    virtual ggml_tensor* view_4d(ggml_tensor* tensor, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, size_t nb1, size_t nb2, size_t nb3, size_t offset) = 0;

    // Repeat / broadcast
    virtual ggml_tensor* repeat(ggml_tensor* tensor, ggml_tensor* target) = 0;

    // Concatenation
    virtual ggml_tensor* concat(ggml_tensor* a, ggml_tensor* b, int dim) = 0;

    // Reduction
    virtual ggml_tensor* sum_rows(ggml_tensor* tensor) = 0;
    virtual ggml_tensor* silu(ggml_tensor* tensor) = 0;
};

// ============================================================================
// Distributions
// ============================================================================

constexpr int kNoAxis = -1;
constexpr double kInf = 1e30;

enum class DistType { R, S, P };

struct Dist {
    DistType type = DistType::R;
    int axis = kNoAxis;

    bool operator==(const Dist& o) const { return type == o.type && axis == o.axis; }
    bool operator!=(const Dist& o) const { return !(*this == o); }
    bool operator<(const Dist& o) const {
        if (type != o.type) return type < o.type;
        return axis < o.axis;
    }
};

constexpr Dist rep() { return Dist{}; }
constexpr Dist shard(int axis) { return {DistType::S, axis}; }
constexpr Dist partial(int axis = kNoAxis) { return {DistType::P, axis}; }

std::string dist_to_string(const Dist& d) {
    switch (d.type) {
        case DistType::R: return "R";
        case DistType::S: return "S(" + std::to_string(d.axis) + ")";
        case DistType::P: return d.axis == kNoAxis ? "P" : "P(" + std::to_string(d.axis) + ")";
    }
    return "?";
}

// The plan -> GGML mapping for the axis field of the split state.
enum ggml_backend_meta_split_axis dist_to_split_axis(const Dist& d) {
    switch (d.type) {
        case DistType::R: return GGML_BACKEND_SPLIT_AXIS_MIRRORED;
        case DistType::S: return static_cast<enum ggml_backend_meta_split_axis>(d.axis);
        case DistType::P: return GGML_BACKEND_SPLIT_AXIS_PARTIAL;
    }
    return GGML_BACKEND_SPLIT_AXIS_NONE;
}

// ============================================================================
// Planner data structures
// ============================================================================

// One way an op can compute: the distribution it produces, the distributions
// its inputs must be in (one per trace input, in order), and the compute cost.
struct Candidate {
    Dist output;
    std::vector<Dist> inputs;
    double comp_cost = 0.0;
};

struct TraceNode {
    int id = 0;
    std::string op_name;
    int rank = 0;                            // 0..4, logical dims
    int64_t ne[4] = {1, 1, 1, 1};            // GGML order, padded with 1s
    std::vector<int> inputs;                 // trace node ids
    std::vector<Candidate> candidates;       // empty = the meta backend cannot run this op
    bool is_param = false;                   // static tensor: R or S(a) storage decision
    bool is_fixed = false;                   // graph input / compute leaf: externally fixed to R
};

// F(node, d): node produces exactly d.
struct ExactState {
    bool done = false;
    double cost = kInf;
    int cand = -1;
    std::vector<Dist> in_dists;
};

// G(node, d): node satisfies d (produces some d' and bridges d' -> d).
struct BestState {
    bool done = false;
    double cost = kInf;
    Dist produced;
};

struct PlanNode {
    int id = 0;
    std::string op_name;
    std::string tensor_name;        // non-empty for params (the callback key)
    Dist produced;
    Dist required;
    std::string bridge;                 // collective between produced and required
    double bridge_cost = 0.0;
};

struct Plan {
    double total_cost = 0.0;
    size_t device_count = 0;   // for printing the per-device split sizes
    bool infeasible = false;
    std::string infeasible_reason;
    std::vector<PlanNode> nodes;        // DFS preorder; printed in reverse = execution order
    std::map<int, Dist> callback_dists; // param node id -> storage distribution
    // The plan -> GGML tensor split mapping: for every statically allocated
    // tensor, the split state a ggml_backend_meta_get_split_state_t callback
    // must return, keyed by tensor name. Compute tensors need no entry: the
    // meta backend derives their splits from these and its per-op rules.
    std::map<std::string, ggml_backend_meta_split_state> callback_states;

    std::string to_string() const {
        std::ostringstream ss;
        if (infeasible) {
            ss << "=== plan INFEASIBLE ===\n";
            ss << "  " << infeasible_reason << "\n";
            ss << "=======================================\n";
            return ss.str();
        }
        ss << "=== plan (total cost " << std::fixed << std::setprecision(2) << total_cost << ") ===\n";
        for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
            const PlanNode& pn = *it;
            ss << "  [" << pn.id << "] " << pn.op_name << (pn.tensor_name.empty() ? "" : " " + pn.tensor_name) << ": ";
            if (pn.produced == pn.required) {
                ss << dist_to_string(pn.produced);
            } else {
                ss << dist_to_string(pn.produced) << " --" << pn.bridge << "--> " << dist_to_string(pn.required);
            }
            ss << "\n";
        }
        return ss.str();
    }
};

// ============================================================================
// PlannerEngine
// ============================================================================

class PlannerEngine : public Engine {
public:
    PlannerEngine(int device_count, double w_comm, double w_comp, double w_mem)
        : device_count_(device_count < 1 ? 1 : device_count),
          w_comm_(w_comm), w_comp_(w_comp), w_mem_(w_mem) {}

    // Solve the DP and reconstruct the plan. The root defaults to the last
    // traced op (the graph output); the required dist defaults to R because
    // the final result must be usable on every device.
    Plan finalize(int root = -1, Dist required = rep());

    // Re-derive every node's split state the way the meta backend does, from
    // the plan's callback table (params) and R (fixed inputs), applying the
    // same per-op rules, and compare against the planned states. Returns
    // false (with an explanation) if the plan is not what the meta device
    // will derive.
    bool verify(const Plan& plan, std::string& error) const;

    // Debug dump of the traced graph: every node with its shape and the
    // output distributions its candidates can produce.
    std::string dump_trace() const;

    // ---------------------------------------------------------------------
    // Engine: tensor creation / initialization
    // ---------------------------------------------------------------------
    ggml_tensor* new_tensor(ggml_context*, ggml_type type, int n_dims, const int64_t* ne) override {
        const int rank = std::clamp(n_dims, 0, 4);
        int64_t padded[4] = {1, 1, 1, 1};
        for (int i = 0; i < rank; ++i)
            padded[i] = ne[i];

        const int id = trace_op("param", {}, param_candidates(rank), rank, padded);
        nodes_[id].is_param = true;
        ggml_tensor* t = make_tensor(id);
        t->type = type;
        return t;
    }

    ggml_tensor* new_tensor_1d(ggml_context* ctx, ggml_type type, int64_t ne0) override {
        const int64_t ne[1] = {ne0};
        return new_tensor(ctx, type, 1, ne);
    }

    // A tensor whose state is externally fixed: a graph input or a compute
    // leaf (Tensor::empty). The meta backend stores compute leaves in the
    // compute buffer as GGML_OP_NONE, which is MIRRORED -- so the only
    // legal fixed state is R.
    void set_input(ggml_tensor* t) override {
        TraceNode& n = nodes_[get_id(t)];
        n.is_fixed = true;
        n.op_name = "input";   // a compute leaf, not a model param
        n.is_param = false;
        n.candidates = {{rep(), {}, 0.0}};
    }

    ggml_tensor* fill(ggml_tensor* t, float) override {
        // The meta runs FILL through handle_generic (state carries over from
        // the shape template); each device simply fills its own slice.
        const int id = get_id(t);
        const int rank = rank_of(t);
        std::vector<Candidate> cands = {{rep(), {rep()}, 0.0}};
        for (int a = 0; a < rank; ++a)
            cands.push_back({shard(a), {shard(a)}, 0.0});
        return make_tensor(trace_op("fill", {id}, std::move(cands), rank, nodes_[id].ne));
    }

    // ---------------------------------------------------------------------
    // Engine: copy / cast
    // ---------------------------------------------------------------------
    ggml_tensor* cont(ggml_tensor* t) override {
        // GGML_OP_CONT and GGML_OP_RESHAPE share the meta's handle_reshape rule.
        const int id = get_id(t);
        return reinterpret_op("cont", id, nodes_[id].ne);
    }

    ggml_tensor* dup(ggml_tensor* t) override {
        // The meta runs DUP with scalar_only: a DUP of a sharded tensor
        // ABORTS. Only a replicated copy is planned.
        const int id = get_id(t);
        return make_tensor(trace_op("dup", {id}, {{rep(), {rep()}, w_comp_}}, nodes_[id].rank, nodes_[id].ne));
    }

    ggml_tensor* cast(ggml_tensor* t, ggml_type) override {
        // GGML_OP_CAST is not in the meta backend's switch -> it ABORTs on
        // any split state. The node gets no candidates: any graph that casts
        // is infeasible (keep the graph dtype-uniform instead).
        const int id = get_id(t);
        return make_tensor(trace_op("cast", {id}, {}, nodes_[id].rank, nodes_[id].ne));
    }

    ggml_tensor* cpy(ggml_tensor* src, ggml_tensor* dst) override {
        // dst is a shape template; data flows from src. The meta routes a
        // sharded CPY through handle_reshape (the shard remaps to the dst
        // shape); a replicated CPY is a plain copy. handle_reshape requires
        // src_rank <= dst_rank, so sharded candidates only exist then.
        const int si = get_id(src);
        const int di = get_id(dst);
        const int rank = nodes_[di].rank;
        std::vector<Candidate> cands = {{rep(), {rep(), rep()}, w_comp_}};
        if (nodes_[si].rank <= rank) {
            for (int a = 0; a < nodes_[si].rank; ++a) {
                if (derive_reshape(nodes_[si].ne, nodes_[di].ne, shard(a)))
                    cands.push_back({shard(derive_reshape_axis(nodes_[si].ne, nodes_[di].ne, a)), {shard(a), rep()}, sharded_comp()});
            }
        }
        return make_tensor(trace_op("cpy", {si, di}, std::move(cands), rank, nodes_[di].ne));
    }

    // ---------------------------------------------------------------------
    // Engine: unary arithmetic
    // ---------------------------------------------------------------------
    ggml_tensor* neg(ggml_tensor* t) override { return unsupported_op("neg", t); }
    ggml_tensor* abs(ggml_tensor* t) override { return unsupported_op("abs", t); }
    ggml_tensor* sqrt(ggml_tensor* t) override { return carry_over_op("sqrt", t, w_comp_); }
    ggml_tensor* exp(ggml_tensor* t) override { return unsupported_op("exp", t); }
    ggml_tensor* log(ggml_tensor* t) override { return carry_over_op("log", t, w_comp_); }
    ggml_tensor* sin(ggml_tensor* t) override { return carry_over_op("sin", t, w_comp_); }
    ggml_tensor* cos(ggml_tensor* t) override { return carry_over_op("cos", t, w_comp_); }

    // ---------------------------------------------------------------------
    // Engine: binary arithmetic
    // ---------------------------------------------------------------------
    ggml_tensor* add(ggml_tensor* l, ggml_tensor* r) override { return binary_op("add", l, r); }
    ggml_tensor* sub(ggml_tensor* l, ggml_tensor* r) override { return binary_op("sub", l, r); }
    ggml_tensor* mul(ggml_tensor* l, ggml_tensor* r) override { return binary_op("mul", l, r); }
    ggml_tensor* div(ggml_tensor* l, ggml_tensor* r) override { return binary_op("div", l, r); }

    // ---------------------------------------------------------------------
    // Engine: scalar arithmetic
    // ---------------------------------------------------------------------
    ggml_tensor* scale(ggml_tensor* t, float) override { return carry_over_op("scale", t, w_comp_); }
    ggml_tensor* clamp(ggml_tensor* t, float, float) override { return carry_over_op("clamp", t, w_comp_); }

    // ---------------------------------------------------------------------
    // Engine: matrix operations
    // ---------------------------------------------------------------------
    ggml_tensor* mul_mat(ggml_tensor* l, ggml_tensor* r) override {
        // Convention (see nn/Linear): lhs = weight [in, out], rhs = activation.
        // ggml_mul_mat: result ne = {lhs->ne[1], rhs->ne[1], rhs->ne[2], rhs->ne[3]},
        // so the batch dims (and the output rank) come from the rhs.
        const int li = get_id(l);
        const int ri = get_id(r);
        const TraceNode& w = nodes_[li];
        const TraceNode& a = nodes_[ri];
        const int64_t out_ne[4] = {w.ne[1], a.ne[1], a.ne[2], a.ne[3]};
        return make_tensor(trace_op("mul_mat", {li, ri}, mul_mat_candidates(w, a), a.rank, out_ne));
    }

    // ---------------------------------------------------------------------
    // Engine: reshape / permute / views
    // ---------------------------------------------------------------------
    ggml_tensor* reshape_1d(ggml_tensor* t, int64_t ne0) override {
        const int64_t out_ne[4] = {ne0, 1, 1, 1};
        return reinterpret_op("reshape", get_id(t), out_ne);
    }
    ggml_tensor* reshape_2d(ggml_tensor* t, int64_t ne0, int64_t ne1) override {
        const int64_t out_ne[4] = {ne0, ne1, 1, 1};
        return reinterpret_op("reshape", get_id(t), out_ne);
    }
    ggml_tensor* reshape_3d(ggml_tensor* t, int64_t ne0, int64_t ne1, int64_t ne2) override {
        const int64_t out_ne[4] = {ne0, ne1, ne2, 1};
        return reinterpret_op("reshape", get_id(t), out_ne);
    }
    ggml_tensor* reshape_4d(ggml_tensor* t, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) override {
        const int64_t out_ne[4] = {ne0, ne1, ne2, ne3};
        return reinterpret_op("reshape", get_id(t), out_ne);
    }

    ggml_tensor* permute(ggml_tensor* t, int a0, int a1, int a2, int a3) override {
        const int id = get_id(t);
        const TraceNode& src = nodes_[id];
        const int ax[4] = {a0, a1, a2, a3};
        int64_t out_ne[4] = {1, 1, 1, 1};
        for (int i = 0; i < 4; ++i)
            out_ne[i] = src.ne[ax[i] >= 0 && ax[i] < 4 ? ax[i] : i];

        // GGML_OP_PERMUTE: a shard of the input along axis b reappears on
        // the output axis i with ax[i] == b (the meta's handle_permute).
        std::vector<Candidate> cands = {{rep(), {rep()}, 0.0}};
        for (int b = 0; b < src.rank; ++b) {
            for (int i = 0; i < src.rank; ++i) {
                if (ax[i] == b)
                    cands.push_back({shard(i), {shard(b)}, 0.0});
            }
        }
        return make_tensor(trace_op("permute", {id}, std::move(cands), src.rank, out_ne));
    }

    ggml_tensor* view_1d(ggml_tensor* t, int64_t ne0, size_t) override {
        const int64_t out_ne[4] = {ne0, 1, 1, 1};
        return reinterpret_op("view", get_id(t), out_ne);
    }
    ggml_tensor* view_2d(ggml_tensor* t, int64_t ne0, int64_t ne1, size_t, size_t) override {
        const int64_t out_ne[4] = {ne0, ne1, 1, 1};
        return reinterpret_op("view", get_id(t), out_ne);
    }
    ggml_tensor* view_3d(ggml_tensor* t, int64_t ne0, int64_t ne1, int64_t ne2, size_t, size_t, size_t) override {
        const int64_t out_ne[4] = {ne0, ne1, ne2, 1};
        return reinterpret_op("view", get_id(t), out_ne);
    }
    ggml_tensor* view_4d(ggml_tensor* t, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, size_t, size_t, size_t, size_t) override {
        const int64_t out_ne[4] = {ne0, ne1, ne2, ne3};
        return reinterpret_op("view", get_id(t), out_ne);
    }

    // ---------------------------------------------------------------------
    // Engine: repeat / broadcast, concatenation
    // ---------------------------------------------------------------------
    ggml_tensor* repeat(ggml_tensor* t, ggml_tensor* target) override {
        // The meta runs REPEAT through handle_generic: all srcs must be in
        // the SAME state. The target is a compute leaf (fixed R), so only a
        // replicated source can be broadcast.
        const int ti = get_id(t);
        const int ri = get_id(target);
        return make_tensor(trace_op("repeat", {ti, ri}, {{rep(), {rep(), rep()}, w_comp_}}, nodes_[ri].rank, nodes_[ri].ne));
    }

    ggml_tensor* concat(ggml_tensor* a, ggml_tensor* b, int dim) override {
        // GGML_OP_CONCAT (the meta's handle_concat): one operand may be
        // sharded along any axis OTHER than the concat dim; both may be
        // sharded along the same axis. (dim is the GGML axis.)
        const int ai = get_id(a);
        const int bi = get_id(b);
        const int rank = std::max(nodes_[ai].rank, nodes_[bi].rank);
        int64_t out_ne[4];
        for (int i = 0; i < 4; ++i)
            out_ne[i] = (i == dim) ? nodes_[ai].ne[i] + nodes_[bi].ne[i] : nodes_[ai].ne[i];

        std::vector<Candidate> cands = {{rep(), {rep(), rep()}, w_comp_}};
        for (int a = 0; a < rank && a != dim; ++a) {
            cands.push_back({shard(a), {shard(a), shard(a)}, sharded_comp()});
            cands.push_back({shard(a), {shard(a), rep()}, sharded_comp()});
            cands.push_back({shard(a), {rep(), shard(a)}, sharded_comp()});
        }
        return make_tensor(trace_op("concat", {ai, bi}, std::move(cands), rank, out_ne));
    }

    // ---------------------------------------------------------------------
    // Engine: reduction
    // ---------------------------------------------------------------------
    ggml_tensor* sum_rows(ggml_tensor* t) override {
        // GGML_OP_SUM_ROWS (the meta's handle_per_row): asserts the src is
        // not sharded along the reduced axis 0 and keeps the src state
        // UNCHANGED (axis preserved, ne preserved).
        const int id = get_id(t);
        const int in_rank = nodes_[id].rank;
        int64_t out_ne[4] = {1, nodes_[id].ne[1], nodes_[id].ne[2], nodes_[id].ne[3]};
        std::vector<Candidate> cands = {{rep(), {rep()}, w_comp_}};
        for (int a = 1; a < in_rank; ++a)
            cands.push_back({shard(a), {shard(a)}, sharded_comp()});
        return make_tensor(trace_op("sum_rows", {id}, std::move(cands), in_rank > 0 ? in_rank - 1 : 0, out_ne));
    }

    ggml_tensor* silu(ggml_tensor* t) override {
        // GGML_OP_SILU is not in the meta backend's switch -> GGML_ABORT
        // ("ggml op not implemented"). No candidates: a graph that uses silu
        // is infeasible on the meta device. Use a supported activation
        // instead (clamp-based ReLU, leaky relu, ggml_unary ops, ...).
        return unsupported_op("silu", t);
    }

private:
    // ---------------------------------------------------------------------
    // Cost model
    // ---------------------------------------------------------------------
    double sharded_comp() const { return w_comp_ / device_count_; }
    double comm_factor() const { return (device_count_ - 1.0) / device_count_; }

    struct Bridge {
        std::string name;
        double cost;
    };

    // The collective needed to turn a tensor in `from` into the distribution
    // `to`, and its per-device cost. The meta backend's only collective is
    // the AllReduce at a PARTIAL subgraph boundary -- there is no
    // AllGather/ReduceScatter/AllToAll, so everything except P -> R is
    // infeasible. A sharded tensor is consumed sharded through the per-op
    // rules; a full tensor is (re-)produced by a row-parallel mul_mat +
    // the implicit AllReduce.
    Bridge bridge(const Dist& from, const Dist& to) const {
        if (from == to) return {"None", 0.0};
        if (from.type == DistType::P && to.type == DistType::R)
            return {"AllReduce", 0.5 * w_comm_ * comm_factor()};
        return {"Infeasible", kInf};
    }

    // ---------------------------------------------------------------------
    // Candidate generation -- exactly the states the meta backend accepts
    // (see the per-op rules in the file header)
    // ---------------------------------------------------------------------
    std::vector<Candidate> param_candidates(int rank) const {
        std::vector<Candidate> cands;
        cands.push_back({rep(), {}, device_count_ * w_mem_});   // full replica on every device
        for (int a = 0; a < rank; ++a)
            cands.push_back({shard(a), {}, w_mem_});            // the weight split across devices
        return cands;
    }

    // Elementwise unary: the meta carries the src state over unchanged.
    std::vector<Candidate> carry_over_candidates(int rank, double cost) const {
        std::vector<Candidate> cands;
        cands.push_back({rep(), {rep()}, cost});
        for (int a = 0; a < rank; ++a)
            cands.push_back({shard(a), {shard(a)}, cost / device_count_});
        return cands;
    }

    // ggml binary op: lhs broadcasts rhs against itself (the project's
    // Tensor operators keep the broadcast superset on the left).
    std::vector<Candidate> binary_candidates(const TraceNode& lhs, const TraceNode& rhs, int out_rank) const {
        std::vector<Candidate> cands;
        cands.push_back({rep(), {rep(), rep()}, w_comp_});
        for (int a = 0; a < out_rank && a < lhs.rank; ++a) {
            if (a < rhs.rank)
                cands.push_back({shard(a), {shard(a), shard(a)}, sharded_comp()});
            // The 2nd operand's dim a is size 1: it is a broadcast (the
            // meta's handle_bin_bcast keeps the 1st operand's shard).
            if (rhs.ne[a] == 1)
                cands.push_back({shard(a), {shard(a), rep()}, sharded_comp()});
        }
        return cands;
    }

    // mul_mat(lhs = weight [in, out], rhs = activation), result rank = rank(rhs).
    // The meta's handle_mul_mat accepts exactly these four tuples; the
    // row-parallel one additionally GGML_ASSERTs that the weight and
    // activation splits are equal, which holds for near-uniform splits iff
    // the contract dim sizes match.
    std::vector<Candidate> mul_mat_candidates(const TraceNode& w, const TraceNode& a) const {
        std::vector<Candidate> cands;
        cands.push_back({rep(), {rep(), rep()}, w_comp_});
        if (w.rank >= 2)
            cands.push_back({shard(0), {shard(1), rep()}, sharded_comp()});    // column-parallel
        if (a.rank >= 2)
            cands.push_back({shard(1), {rep(), shard(1)}, sharded_comp()});    // token-parallel
        if (w.rank >= 1 && a.rank >= 1 && w.ne[0] == a.ne[0])
            cands.push_back({partial(), {shard(0), shard(0)}, sharded_comp()}); // row-parallel
        return cands;
    }

    // ---------------------------------------------------------------------
    // The meta's handle_reshape, ported: given an input sharded along axis
    // `axis`, which output axis does a reshape/view/cont to shape
    // `out_ne` produce? Returns std::nullopt when the meta would
    // GGML_ABORT("shape mismatch ...").
    // ---------------------------------------------------------------------
    static int ggml_n_dims(const int64_t ne[4]) {
        int n = 4;
        while (n > 1 && ne[n - 1] == 1) --n;
        return n;
    }

    static bool derive_reshape(const int64_t src_ne[4], const int64_t out_ne[4], const Dist& in) {
        if (in.type == DistType::R)
            return true;
        if (in.type != DistType::S)
            return false;   // a P source is visible as R (handled by the R candidate)
        const int axis = in.axis;
        if (axis < 0 || axis > 3)
            return false;
        // The nr[0] == 1 fast path (the planner only produces single-segment
        // splits with nr == 1): a shard of the src's last meaningful dim
        // lands on the output's last meaningful dim.
        if (axis == ggml_n_dims(src_ne) - 1)
            return true;
        int64_t base_ne_in = 1;
        for (int dim = 0; dim <= axis; ++dim)
            base_ne_in *= src_ne[dim];
        int64_t base_ne_out = 1;
        for (int dim = 0; dim < 4; ++dim) {
            base_ne_out *= out_ne[dim];
            if (base_ne_out % base_ne_in == 0)
                return true;
            if (base_ne_out > base_ne_in)
                return true;   // the meta asserts n_segments == 1 && nr[0] == 1 (both true here)
        }
        return false;   // the meta GGML_ABORTs: "shape mismatch"
    }

    // reshape/cont/view: zero-cost memory reinterpretation.
    ggml_tensor* reinterpret_op(const char* name, int id, const int64_t out_ne[4]) {
        const TraceNode& src = nodes_[id];
        std::vector<Candidate> cands = {{rep(), {rep()}, 0.0}};
        for (int a = 0; a < src.rank; ++a) {
            if (derive_reshape(src.ne, out_ne, shard(a)))
                cands.push_back({shard(derive_reshape_axis(src.ne, out_ne, a)), {shard(a)}, 0.0});
        }
        return make_tensor(trace_op(name, {id}, std::move(cands), out_rank_of(out_ne), out_ne));
    }

    // Which output axis a shard of input axis `a` maps to (handle_reshape).
    static int derive_reshape_axis(const int64_t src_ne[4], const int64_t out_ne[4], int a) {
        if (a == ggml_n_dims(src_ne) - 1)
            return ggml_n_dims(out_ne) - 1;
        int64_t base_ne_in = 1;
        for (int dim = 0; dim <= a; ++dim)
            base_ne_in *= src_ne[dim];
        int64_t base_ne_out = 1;
        for (int dim = 0; dim < 4; ++dim) {
            base_ne_out *= out_ne[dim];
            if (base_ne_out % base_ne_in == 0)
                return dim;
            if (base_ne_out > base_ne_in)
                return dim;
        }
        return a;   // unreachable for candidate axes (derive_reshape checked)
    }

    static int out_rank_of(const int64_t ne[4]) { return ggml_n_dims(ne); }

    ggml_tensor* carry_over_op(const char* name, ggml_tensor* t, double cost) {
        const int id = get_id(t);
        return make_tensor(trace_op(name, {id}, carry_over_candidates(nodes_[id].rank, cost), nodes_[id].rank, nodes_[id].ne));
    }

    // An op the meta backend does not handle: zero candidates, so the DP
    // can only report the graph infeasible (with this op named).
    ggml_tensor* unsupported_op(const char* name, ggml_tensor* t) {
        const int id = get_id(t);
        return make_tensor(trace_op(name, {id}, {}, nodes_[id].rank, nodes_[id].ne));
    }

    // ---------------------------------------------------------------------
    // Tracing helpers
    // ---------------------------------------------------------------------
    int rank_of(ggml_tensor* t) const { return nodes_[get_id(t)].rank; }

    int trace_op(const char* name, const std::vector<int>& inputs,
                 std::vector<Candidate> candidates, int rank, const int64_t ne[4]) {
        const int id = (int)nodes_.size();
        TraceNode n;
        n.id = id;
        n.op_name = name;
        n.rank = rank;
        for (int i = 0; i < 4; ++i)
            n.ne[i] = ne[i];
        n.inputs = inputs;
        n.candidates = std::move(candidates);
        nodes_.push_back(std::move(n));
        return id;
    }

    ggml_tensor* make_tensor(int id) {
        auto* raw = new ggml_tensor();
        tensors_.emplace_back(raw);
        raw_of_.push_back(raw);
        raw_to_id_[raw] = id;
        return raw;
    }

    int get_id(ggml_tensor* t) const { return raw_to_id_.at(t); }

    ggml_tensor* binary_op(const char* name, ggml_tensor* l, ggml_tensor* r) {
        const int li = get_id(l);
        const int ri = get_id(r);
        const int rank = std::max(nodes_[li].rank, nodes_[ri].rank);
        int64_t out_ne[4];
        for (int i = 0; i < 4; ++i)
            out_ne[i] = std::max(nodes_[li].ne[i], nodes_[ri].ne[i]);
        return make_tensor(trace_op(name, {li, ri}, binary_candidates(nodes_[li], nodes_[ri], rank), rank, out_ne));
    }

    // ---------------------------------------------------------------------
    // Plan -> GGML split mapping
    // ---------------------------------------------------------------------
    // Materialize the split state a callback must return for a static
    // tensor with distribution `d`, GGML shape `ne` and dtype `type`:
    //   R  -> the canonical MIRRORED form (axis = MIRRORED, ne = 0,
    //         nr[0] = 1, n_segments = 1; see llama.cpp's get_tensor_split)
    //   S(a) -> one segment, nr = 1, near-uniform per-device sizes with
    //         llama.cpp's even-split boundaries (boundary(i) = ne * i / n);
    //         for a == 0 the boundaries are additionally rounded down to
    //         multiples of ggml_blck_size (the meta GGML_ASSERTs it)
    // P is never materialized: the callback is only called for static
    // tensors, and a static tensor is never PARTIAL.
    ggml_backend_meta_split_state materialize(const Dist& d, const int64_t ne[4], ggml_type type) const {
        ggml_backend_meta_split_state st;
        std::memset(&st, 0, sizeof(st));
        st.axis = dist_to_split_axis(d);
        st.nr[0] = 1;
        st.n_segments = 1;
        if (d.type == DistType::S) {
            const int64_t gran = d.axis == 0 ? ggml_blck_size(type) : 1;
            int64_t low = 0;
            for (int j = 0; j < device_count_; ++j) {
                int64_t high = ne[d.axis] * (int64_t)(j + 1) / device_count_;
                if (j + 1 < device_count_)
                    high = (high / gran) * gran;
                st.ne[j] = high - low;
                low = high;
            }
        }
        return st;
    }

    std::string param_name(int id) const {
        const char* n = raw_of_[id]->name;
        return n[0] ? n : ("node" + std::to_string(id));
    }

    // ---------------------------------------------------------------------
    // Dynamic program
    // ---------------------------------------------------------------------
    // F(node, d): node produces exactly d.
    ExactState& exact(int node, const Dist& d) {
        auto& m = exact_memo_[node][d];
        if (m.done) return m;
        m.done = true;

        const TraceNode& n = nodes_[node];
        for (int c = 0; c < (int)n.candidates.size(); ++c) {
            const Candidate& cand = n.candidates[c];
            if (cand.output != d) continue;

            double cost = cand.comp_cost;
            bool ok = true;
            std::vector<Dist> ins;
            ins.reserve(cand.inputs.size());
            for (size_t i = 0; i < cand.inputs.size(); ++i) {
                const double in_cost = best(n.inputs[i], cand.inputs[i]).cost;
                if (in_cost >= kInf / 2) { ok = false; break; }
                cost += in_cost;
                ins.push_back(cand.inputs[i]);
            }
            if (ok && cost < m.cost)
                m = {true, cost, c, std::move(ins)};
        }
        return m;
    }

    // G(node, d): node satisfies d -- produce some producible d', then bridge.
    BestState& best(int node, const Dist& d) {
        auto& m = best_memo_[node][d];
        if (m.done) return m;
        m.done = true;

        std::set<Dist> producible;
        for (const Candidate& cand : nodes_[node].candidates)
            producible.insert(cand.output);

        for (const Dist& p : producible) {
            const double exact_cost = exact(node, p).cost;
            if (exact_cost >= kInf / 2) continue;
            const Bridge b = bridge(p, d);
            if (b.cost >= kInf / 2) continue;
            const double total = exact_cost + b.cost;
            if (total < m.cost)
                m = {true, total, p};
        }
        return m;
    }

    void emit(int node, const Dist& required, Plan& plan, std::set<std::pair<int, Dist>>& emitted) {
        // A tensor consumed several times in the same distribution is planned
        // once. (A tensor needed in two different distributions is planned per
        // distribution; finalize() then rejects the plan, because the meta
        // backend derives exactly one state per tensor.)
        if (!emitted.insert({node, required}).second) return;

        const BestState& b = best(node, required);
        const Bridge br = bridge(b.produced, required);

        PlanNode pn;
        pn.id = node;
        pn.op_name = nodes_[node].op_name;
        pn.tensor_name = nodes_[node].is_param ? param_name(node) : "";
        pn.produced = b.produced;
        pn.required = required;
        pn.bridge = std::move(br.name);
        pn.bridge_cost = br.cost;
        plan.nodes.push_back(std::move(pn));

        const ExactState& e = exact(node, b.produced);
        if (e.cand < 0) return;
        const Candidate& cand = nodes_[node].candidates[e.cand];
        const TraceNode& n = nodes_[node];
        for (size_t i = 0; i < cand.inputs.size(); ++i)
            emit(n.inputs[i], cand.inputs[i], plan, emitted);
    }

    std::string infeasibility_reason(const Dist& required) const {
        std::string r;
        for (const TraceNode& n : nodes_) {
            if (n.is_fixed || !n.candidates.empty()) continue;
            if (!r.empty()) r += "; ";
            r += n.op_name + " (node " + std::to_string(n.id) + ") is not supported by the meta backend (no split-state rule)";
        }
        if (r.empty())
            r = "no feasible split plan satisfies the required output distribution " + dist_to_string(required);
        return r;
    }

    // ---------------------------------------------------------------------
    // State
    // ---------------------------------------------------------------------
    int device_count_;
    double w_comm_, w_comp_, w_mem_;

    std::vector<TraceNode> nodes_;
    std::vector<std::unique_ptr<ggml_tensor>> tensors_;
    std::vector<ggml_tensor*> raw_of_;                 // index = trace node id
    std::unordered_map<ggml_tensor*, int> raw_to_id_;

    std::map<int, std::map<Dist, ExactState>> exact_memo_;
    std::map<int, std::map<Dist, BestState>> best_memo_;
};

Plan PlannerEngine::finalize(int root, Dist required) {
    exact_memo_.clear();
    best_memo_.clear();

    Plan plan;
    if (nodes_.empty()) return plan;
    if (root < 0) root = (int)nodes_.size() - 1;   // the last traced op is the output

    plan.device_count = device_count_;

    const double total = best(root, required).cost;
    if (total >= kInf / 2) {
        plan.infeasible = true;
        plan.infeasible_reason = infeasibility_reason(required);
        return plan;
    }
    plan.total_cost = total;

    std::set<std::pair<int, Dist>> emitted;
    emit(root, required, plan, emitted);

    for (const PlanNode& pn : plan.nodes)
        if (nodes_[pn.id].is_param)
            plan.callback_dists[pn.id] = pn.produced;

    // The meta backend derives exactly one split state per tensor, so a
    // tensor consumed in two different distributions cannot be planned
    // (one static storage layout / one compute layout per tensor).
    std::map<int, Dist> single_state;
    for (const PlanNode& pn : plan.nodes) {
        auto [it, inserted] = single_state.insert({pn.id, pn.produced});
        if (!inserted && it->second != pn.produced) {
            plan.infeasible = true;
            plan.infeasible_reason = nodes_[pn.id].op_name + " (node " + std::to_string(pn.id) +
                ") is required in both " + dist_to_string(it->second) + " and " +
                dist_to_string(pn.produced) + ", but the meta backend derives a single state per tensor";
            return plan;
        }
    }

    // The plan -> GGML tensor split mapping.
    for (const auto& [id, dist] : plan.callback_dists)
        plan.callback_states[param_name(id)] = materialize(dist, nodes_[id].ne, raw_of_[id]->type);
    return plan;
}

bool PlannerEngine::verify(const Plan& plan, std::string& error) const {
    if (plan.infeasible) {
        error = plan.infeasible_reason;
        return false;
    }

    // Re-derive every node's split state the way the meta backend does
    // (ggml-backend-meta.cpp: the callback states for static tensors,
    // GGML_OP_NONE = MIRRORED for compute leaves, and the per-op rules),
    // then compare against the planned states. The rules are exactly the
    // node candidates, so a mismatch means the DP/emit drifted from what
    // the meta device will actually derive.
    std::map<int, Dist> planned;
    for (const PlanNode& pn : plan.nodes)
        planned[pn.id] = pn.produced;

    std::vector<Dist> visible(nodes_.size());
    for (int id = 0; id < (int)nodes_.size(); ++id) {
        const TraceNode& n = nodes_[id];
        Dist d;
        if (n.is_fixed) {
            d = rep();   // compute buffer, GGML_OP_NONE
        } else if (n.is_param) {
            const auto it = plan.callback_dists.find(id);
            if (it == plan.callback_dists.end()) {
                error = param_name(id) + " (node " + std::to_string(id) +
                    ") has no storage state in the plan's callback table";
                return false;
            }
            d = it->second;
        } else {
            std::vector<Dist> in_states;
            in_states.reserve(n.inputs.size());
            for (const int in : n.inputs)
                in_states.push_back(visible[in]);

            const Candidate* match = nullptr;
            int count = 0;
            for (const Candidate& c : n.candidates) {
                if (c.inputs == in_states) {
                    match = &c;
                    ++count;
                }
            }
            if (count != 1) {
                std::ostringstream ss;
                ss << n.op_name << " (node " << id << "): the meta rules give " << count
                   << " state(s) for input states {";
                for (size_t i = 0; i < in_states.size(); ++i)
                    ss << (i ? ", " : "") << dist_to_string(in_states[i]);
                ss << "}";
                error = ss.str();
                return false;
            }
            d = match->output;
        }

        const auto it = planned.find(id);
        if (it == planned.end()) {
            error = n.op_name + " (node " + std::to_string(id) + ") is missing from the plan";
            return false;
        }
        if (it->second != d) {
            error = n.op_name + " (node " + std::to_string(id) + "): planned " +
                dist_to_string(it->second) + " but the meta backend derives " + dist_to_string(d);
            return false;
        }

        // Consumers of a PARTIAL tensor see MIRRORED: the meta derives
        // source states with assume_sync = true, and the row-parallel
        // mul_mat returns MIRRORED in that mode (the AllReduce happens at
        // the subgraph boundary, before the consumer).
        visible[id] = (d.type == DistType::P) ? rep() : d;
    }
    return true;
}

std::string PlannerEngine::dump_trace() const {
    std::ostringstream ss;
    for (const TraceNode& n : nodes_) {
        ss << "  [" << n.id << "] " << n.op_name;
        if (n.is_fixed) ss << " (fixed R)";
        if (n.is_param) ss << " " << (raw_of_[n.id]->name[0] ? raw_of_[n.id]->name : "?");
        ss << " ne={" << n.ne[0];
        for (int i = 1; i < n.rank; ++i) ss << ", " << n.ne[i];
        ss << "} in={";
        for (size_t i = 0; i < n.inputs.size(); ++i)
            ss << (i ? ", " : "") << n.inputs[i];
        ss << "} candidates:";
        for (const Candidate& c : n.candidates)
            ss << " " << dist_to_string(c.output);
        if (n.candidates.empty())
            ss << " (none -- unsupported by the meta backend)";
        ss << "\n";
    }
    return ss.str();
}

// ============================================================================
// Mock module framework (mirrors src/ggml/Scope.hpp and src/nn:
// Module / Parameter / Visitor; the Tensor mirrors src/ggml/Tensor.cpp so
// the traced graph structure matches what the real forward() produces)
// ============================================================================

class Context {
public:
    ggml_context* ctx_ = nullptr;
    ggml_context* operator*() const { return ctx_; }
};

// Same RAII / static-access pattern as src/ggml/Scope.hpp.
class Scope {
public:
    Scope(Context& context, Engine& engine)
        : previous_context_(current_context_), previous_engine_(current_engine_)
    {
        current_context_ = &context;
        current_engine_ = &engine;
    }
    Scope(const Scope& other)
        : previous_context_(current_context_), previous_engine_(current_engine_)
    {
        current_context_ = other.current_context_;
        current_engine_ = other.current_engine_;
    }
    ~Scope() {
        current_context_ = previous_context_;
        current_engine_ = previous_engine_;
    }

    static Context& context() { return *current_context_; }
    static Engine& engine() { return *current_engine_; }

private:
    inline static thread_local Context* current_context_ = nullptr;
    inline static thread_local Engine* current_engine_ = nullptr;
    Context* previous_context_ = nullptr;
    Engine* previous_engine_ = nullptr;
};

class Tensor {
public:
    // Mirrors the project's Tensor::Shape: dims are given in PyTorch order
    // (first = slowest / outermost) but stored in GGML order (ne_[0] fastest).
    class Shape {
    public:
        static bool broadcasts(const Shape& a, const Shape& b);
        static Shape broadcast(const Shape& lhs, const Shape& rhs);

        Shape(int64_t rank = 0) : ne_({0, 0, 0, 0}), rank_(rank) {}

        Shape(const std::initializer_list<int64_t>& list) : ne_({0, 0, 0, 0}), rank_(list.size()) {
            size_t i = 0;
            for (auto it = std::rbegin(list); it != std::rend(list); ++it)
                ne_[i++] = *it;
        }

        const int64_t& operator [](const int64_t& index) const { return ne_[rank_ - 1 - normalize_index(index)]; }
        int64_t& operator [](const int64_t& index) { return ne_[rank_ - 1 - normalize_index(index)]; }

        const int64_t& rank() const { return rank_; }
        const int64_t* data() const { return ne_.data(); }   // GGML order

        std::string to_string() const;

        bool operator ==(const Shape& other) const { return ne_ == other.ne_ && rank_ == other.rank_; }
        bool operator !=(const Shape& other) const { return !(*this == other); }
    private:
        std::array<int64_t, 4> ne_;
        int64_t rank_;

        int64_t normalize_index(int64_t index) const {
            if (index < 0)
                index += rank_;
            if (index < 0 || index >= rank_)
                throw std::out_of_range("Shape index out of range");
            return index;
        }
    };

    Tensor() = default;
    Tensor(ggml_tensor* t, const Shape& shape, ggml_type dtype = kMockType, bool contiguous = true)
        : t_(t), shape_(shape), dtype_(dtype), contiguous_(contiguous) {}

    Tensor(Tensor&&) = default;
    Tensor& operator=(Tensor&&) = default;
    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;

    // Same as the project: a compute leaf in the current context. The
    // planner marks it as a fixed-R node (the meta stores compute leaves in
    // the compute buffer as MIRRORED / GGML_OP_NONE).
    static Tensor empty(Context& context, const Shape& shape, ggml_type type) {
        auto tensor = shape.rank() == 0
            ? Tensor(Scope::engine().new_tensor_1d(*context, type, 1), shape, type)
            : Tensor(Scope::engine().new_tensor(*context, type, (int)shape.rank(), shape.data()), shape, type);
        Scope::engine().set_input(tensor.t_);
        return tensor;
    }

    ggml_tensor* operator*() const { return t_; }   // the project's Tensor dereferences to ggml_tensor*
    const Shape& shape() const { return shape_; }
    int64_t ndim() const { return shape_.rank(); }
    ggml_type dtype() const { return dtype_; }

    // The mock graph is dtype-uniform: no-op (the project unifies here).
    Tensor to(ggml_type) const { return *this; }

    Tensor contiguous() const {
        return Tensor(Scope::engine().cont(t_), shape_, dtype_, true);
    }
    Tensor clone() const {
        return Tensor(Scope::engine().dup(t_), shape_, dtype_, true);
    }
    Tensor scale(float value) const {
        return Tensor(Scope::engine().scale(t_, value), shape_, dtype_, true);
    }

    // Mirrors the project: ggml_clamp is in-place, so the project clones
    // first. NOTE: that clone (DUP) is not sharding-compatible (the meta
    // aborts on a DUP of a sharded tensor), so sharding-friendly code calls
    // the engine's clamp() directly (see the ReLU demo below).
    Tensor clamp(float a, float b) const {
        auto cloned = clone();
        return Tensor(Scope::engine().clamp(cloned.t_, a, b), cloned.shape_, dtype_, true);
    }

    Tensor operator+(Tensor rhs) const {
        auto lhs = *this;
        auto target = Shape::broadcast(lhs.shape_, rhs.shape_);

        // ggml_add() natively broadcasts its second argument against the
        // first (the first must already be a broadcast superset). Addition
        // is commutative, so use the argument order that lets ggml broadcast.
        if (Shape::broadcasts(lhs.shape_, rhs.shape_))
            return Tensor(Scope::engine().add(lhs.t_, rhs.t_), target, dtype_, true);

        if (Shape::broadcasts(rhs.shape_, lhs.shape_))
            return Tensor(Scope::engine().add(rhs.t_, lhs.t_), target, dtype_, true);

        // Neither shape is a superset of the other: fall back to explicit
        // expansion (the project does the same).
        lhs = lhs.expand(target);
        rhs = rhs.expand(target);
        return Tensor(Scope::engine().add(lhs.t_, rhs.t_), target, dtype_, true);
    }

    Tensor operator*(Tensor rhs) const {
        auto lhs = *this;
        auto target = Shape::broadcast(lhs.shape_, rhs.shape_);

        if (Shape::broadcasts(lhs.shape_, rhs.shape_))
            return Tensor(Scope::engine().mul(lhs.t_, rhs.t_), target, dtype_, true);

        if (Shape::broadcasts(rhs.shape_, lhs.shape_))
            return Tensor(Scope::engine().mul(rhs.t_, lhs.t_), target, dtype_, true);

        lhs = lhs.expand(target);
        rhs = rhs.expand(target);
        return Tensor(Scope::engine().mul(lhs.t_, rhs.t_), target, dtype_, true);
    }

    // Mirrors the project's reshape: infer -1, materialize non-contiguous
    // sources first, dispatch by rank.
    Tensor reshape(const Shape& new_shape) const {
        Shape out(new_shape);

        int64_t infer_dim = -1;
        int64_t known_product = 1;
        for (int64_t i = 0; i < out.rank(); ++i) {
            const int64_t dim = out[i];
            if (dim == -1) {
                if (infer_dim != -1)
                    throw std::invalid_argument("reshape(): only one dimension may be inferred");
                infer_dim = (int64_t)i;
            } else {
                known_product *= dim;
            }
        }

        int64_t numel = 1;
        for (int64_t i = 0; i < shape_.rank(); ++i)
            numel *= shape_[i];

        if (infer_dim != -1)
            out[infer_dim] = numel / known_product;

        auto src = *this;
        if (!contiguous_)
            src = src.contiguous();

        switch (out.rank()) {
            case 0: return Tensor(Scope::engine().reshape_1d(src.t_, 1), out, dtype_, true);
            case 1: return Tensor(Scope::engine().reshape_1d(src.t_, out.data()[0]), out, dtype_, true);
            case 2: return Tensor(Scope::engine().reshape_2d(src.t_, out.data()[0], out.data()[1]), out, dtype_, true);
            case 3: return Tensor(Scope::engine().reshape_3d(src.t_, out.data()[0], out.data()[1], out.data()[2]), out, dtype_, true);
            case 4: return Tensor(Scope::engine().reshape_4d(src.t_, out.data()[0], out.data()[1], out.data()[2], out.data()[3]), out, dtype_, true);
        }
        throw std::invalid_argument("reshape(): unsupported rank");
    }

    Tensor unsqueeze(int64_t dim) const {
        auto rank = ndim();
        dim = dim < 0 ? dim + rank + 1 : dim;   // the project allows the extra +1 insertion position

        Shape out(rank + 1);
        for (auto src = 0, dst = 0; dst < rank + 1; ++dst) {
            if (dst == dim)
                out[dst] = 1;
            else
                out[dst] = shape_[src++];
        }
        return reshape(out);
    }

    // Mirrors the project's expand: pad the rank, then repeat.
    Tensor expand(const Shape& new_shape) const {
        if (new_shape == shape_)
            return *this;
        if (new_shape.rank() == 0)
            return *this;

        auto src = *this;
        while (src.shape().rank() < new_shape.rank())
            src = src.unsqueeze(0);

        Shape repeats(new_shape.rank());
        for (int64_t i = 0; i < new_shape.rank(); ++i) {
            const int64_t current = src.shape()[i];
            const int64_t target = new_shape[i];
            if (current == target)
                repeats[i] = 1;
            else if (current == 1)
                repeats[i] = target;
            else
                throw std::runtime_error("expand(): incompatible dimension");
        }
        return src.repeat(repeats);
    }

    // Mirrors the project's repeat (ggml_repeat): the target tensor
    // describes the output shape.
    Tensor repeat(const Shape& repeats) const {
        const int64_t rank = ndim();
        if (repeats.rank() != rank)
            throw std::invalid_argument("repeat(): number of repeat dimensions must match tensor rank");

        Shape out(rank);
        for (int64_t i = 0; i < rank; ++i)
            out[i] = shape_[i] * repeats[i];

        auto target = empty(Scope::context(), out, dtype());
        return Tensor(Scope::engine().repeat(t_, *target), out, dtype_, true);
    }

    ggml_tensor* t_ = nullptr;
    Shape shape_;
    ggml_type dtype_ = kMockType;
    bool contiguous_ = true;
};

Tensor::Shape Tensor::Shape::broadcast(const Tensor::Shape& lhs, const Tensor::Shape& rhs) {
    const int64_t rank = std::max(lhs.rank(), rhs.rank());
    if (rank == 0)
        return Tensor::Shape();
    if (lhs.rank() == 0)
        return rhs;
    if (rhs.rank() == 0)
        return lhs;

    Shape result(rank);
    for (int64_t i = 0; i < rank; ++i) {
        const int64_t dl = (i < lhs.rank()) ? lhs.data()[i] : 1;
        const int64_t dr = (i < rhs.rank()) ? rhs.data()[i] : 1;
        if (dl == dr) {
            result.ne_[i] = dl;
        } else if (dl == 1) {
            result.ne_[i] = dr;
        } else if (dr == 1) {
            result.ne_[i] = dl;
        } else {
            throw std::invalid_argument("Shapes are not broadcastable.");
        }
    }
    return result;
}

// True if a's shape is a broadcast superset of b's shape (mirrors the
// project's helper): every dim of a is a multiple of the matching dim of
// b (missing dims count as 1).
bool ggml_broadcasts(const Tensor::Shape& a, const Tensor::Shape& b) {
    const int64_t rank = std::max(a.rank(), b.rank());
    for (int64_t i = 0; i < rank; ++i) {
        const int64_t da = (a.rank() == 0 || i >= a.rank()) ? 1 : a.data()[i];
        const int64_t db = (b.rank() == 0 || i >= b.rank()) ? 1 : b.data()[i];
        if (db == 0 || da % db != 0)
            return false;
    }
    return true;
}
std::string Tensor::Shape::to_string() const {
    std::ostringstream oss;
    oss << "(";
    for (int64_t i = 0; i < rank_; ++i)
        oss << (i ? ", " : "") << (*this)[i];
    oss << ")";
    return oss.str();
}


class Parameter;
class Module;

class Visitor {
public:
    virtual void visit(Parameter&, std::vector<std::string>) {}
    virtual void visit(Module&, std::vector<std::string>) {}
};

class Module {
public:
    // Ordered (unlike the project's unordered_map) so the demo prints stable node ids.
    using Children = std::map<std::string, std::shared_ptr<Module>>;

    virtual ~Module() = default;

    virtual void accept(Visitor& visitor, std::vector<std::string> path = {}) {
        for (auto& [name, child] : modules) {
            auto child_path = path;
            child_path.push_back(name);
            child->accept(visitor, std::move(child_path));
        }
        visitor.visit(*this, std::move(path));
    }

protected:
    Children modules;
};

class Parameter : public Module {
public:
    explicit Parameter(Tensor::Shape shape) : shape_(std::move(shape)) {}

    const Tensor::Shape& shape() const { return shape_; }
    void set(Tensor t) { tensor_ = std::move(t); }
    Tensor forward() { return tensor_; }

    void accept(Visitor& visitor, std::vector<std::string> path) override {
        visitor.visit(*this, std::move(path));
    }

private:
    Tensor::Shape shape_;
    Tensor tensor_;
};

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
    // Mirrors the project's nn/SiLU (src/nn/SiLU.hpp). NOTE: GGML_OP_SILU
    // is not handled by the meta backend, so a graph that uses this module
    // is infeasible on the meta device (see demo 2).
    Tensor forward(Scope scope, Tensor x) {
        return Tensor(scope.engine().silu(*x), x.shape());
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

// Stand-in for the GGUF loader: gives every Parameter a tensor of the right
// shape and the dotted name the callback table is keyed by. (Tests use tiny
// random models; the planner does not care about values.)
class CreateRandomParametersVisitor : public Visitor {
public:
    void visit(Parameter& parameter, std::vector<std::string> path) override {
        std::string name;
        for (size_t i = 0; i < path.size(); ++i)
            name += (i ? "." : "") + path[i];

        Tensor::Shape shape = parameter.shape();
        Tensor t = Tensor(Scope::engine().new_tensor(*Scope::context(), kMockType, (int)shape.rank(), shape.data()), shape);
        ggml_set_name(t.t_, name.c_str());
        parameter.set(std::move(t));
    }
};

// ============================================================================
// Demo
// ============================================================================

void print_callback_table(const Plan& plan) {
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
        Context context;
        PlannerEngine planner(/*device_count=*/2, /*w_comm=*/0.5, /*w_comp=*/1.0, /*w_mem=*/0.1);
        Scope scope(context, planner);

        // Graph input: rank-4 activation, PyTorch shape (2, 3, 4, 8)
        // == GGML ne {8, 4, 3, 2}; created like a pipeline input (a
        // compute leaf: fixed R).
        Tensor x = Tensor::empty(context, Tensor::Shape{2, 3, 4, 8}, kMockType);

        MLP<ReLU> model;
        CreateRandomParametersVisitor visitor;
        model.accept(visitor);

        (void)model.forward(scope, x);

        std::cout << "traced graph:\n" << planner.dump_trace() << "\n";
        Plan plan = planner.finalize();
        std::cout << plan.to_string();
        print_callback_table(plan);

        std::string error;
        if (planner.verify(plan, error))
            std::cout << "verification: OK -- the plan matches the meta backend's split-state derivation\n";
        else
            std::cout << "verification FAILED: " << error << "\n";
    }

    {
        // Demo 2: the same MLP with a SiLU activation. GGML_OP_SILU is not
        // in the meta backend's switch, so the planner must report the graph
        // infeasible instead of planning something the meta device would
        // abort on.
        Context context;
        PlannerEngine planner(/*device_count=*/2, /*w_comm=*/0.5, /*w_comp=*/1.0, /*w_mem=*/0.1);
        Scope scope(context, planner);
        Tensor x = Tensor::empty(context, Tensor::Shape{2, 3, 4, 8}, kMockType);

        MLP<SiLU> model;
        CreateRandomParametersVisitor visitor;
        model.accept(visitor);

        (void)model.forward(scope, x);

        std::cout << "\n";
        std::cout << planner.finalize().to_string();
    }

    return 0;
}
