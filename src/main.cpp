// ============================================================================
// Tensor-parallel sharding planner -- standalone proof of concept
//
// This file is intentionally self-contained: it mocks the small slice of the
// project's ggml/nn API surface it needs, so it can be built and run
// directly, without CMake:
//
//     g++ -std=c++17 -O2 -o sharding-poc src/main.cpp && ./sharding-poc
//
// The pieces that will be migrated into the project:
//
//   * ShardingRuntime (drafted in src/ggml/ShardingRuntime.hpp): it
//     implements the exact Runtime interface from src/ggml/Runtime.hpp. While
//     a module's forward() runs through it, it delegates every tensor
//     creation to a parent Runtime (which creates the ggml tensors in a
//     ggml_context, exactly as a real engine would) and records the graph
//     built through it. ONE engine traces everything: every context's
//     forward() runs through the same engine (the scope switches
//     contexts), and a tensor touched by several forwards (a shared
//     parameter) is ONE trace node. Candidate generation is
//     self-contained: the engine is constructed with the cost weights that
//     shape the candidates (w_comp for op compute, w_mem for static
//     storage) and the MetaDevice, whose device count sizes the sharded
//     candidates.
//   * ShardingAllocator (drafted in src/ggml/ShardingAllocator.hpp): the
//     sharded version of the project's Allocator (src/ggml/Allocator.hpp).
//     ONE allocator is aware of ALL the contexts: the persistent weights
//     context and every compute context are registered on it with
//     use(context, device). The allocator OWNS the trace: it constructs
//     the ONE ShardingRuntime every context's forward() runs through
//     (allocator.runtime()), so every context's graph is one subgraph of
//     that single trace. The allocator IS the planning state: the
//     allocation is WHY we plan -- we plan to allocate the tensors
//     optimally across the devices -- and the plan covers the whole
//     trace: a tensor shared by several contexts (typically a parameter
//     in the persistent weights context) is one trace node, planned
//     exactly once, for all of them (the first output to plan a shared
//     parameter decides its split; every later output adapts). The
//     outputs that define the plan's DP goals arrive with each
//     allocate(outputs) call -- the graph's outputs, one goal per
//     output -- and the plan round runs once per distinct output set.
//   * MetaDevice (a mock of src/ggml/MetaDevice.hpp): the shared resource:
//     the split-state callback table, GLOBAL across contexts -- and across
//     every allocator that commits to it. The splits are just GGML's way of
//     doing the sharding; the table is the handoff between the plan and the
//     meta backend. allocate() commits the plan by replacing the table's
//     entries for the allocator's tensors (erase what this allocator
//     traced, insert the new states). split(tensor) returns a tensor's
//     EFFECTIVE state (its planned state, or the canonical MIRRORED
//     default) -- the source of truth the allocator queries when it sizes
//     a slice.
//   * The allocation schema below (Buffer / Allocator): per-context
//     buffers, allocated only when Computation runs -- deferred, so the
//     plan can choose the weights' split before they are allocated into
//     their own buffer -- and aware that they must reallocate when a
//     re-plan changes the splits (the device count): the allocator
//     snapshots the effective split state it allocated every tensor with,
//     and frees + reallocates every context whose snapshots went stale.
//     Because the MetaDevice is GLOBAL (it spans multiple contexts), a
//     changed sharding is exactly this: a split-table update plus the
//     reallocation of whatever the old splits allocated.
//
// Everything else (mock ggml types, Context / Tensor / Scope / Module
// framework, the ContextRuntime parent stand-in, main()) is throwaway
// scaffolding that only exists to exercise the planner without pulling in
// a real backend.
//
// The real usage pattern
// ----------------------
//   1. Load the model: the loader creates the weight tensors in a
//      persistent weights context (unallocated), and every compute context
//      has its own. Create ONE ShardingAllocator over the parent Runtime
//      (it constructs the ShardingRuntime that traces everything) -- the
//      planning boundary, because the allocation is why we plan -- and
//      register every context on it: use(weights_context, meta),
//      use(compute_context, meta), ... Allocate nothing yet.
//   2. Trace every context's forward() through the allocator's engine:
//      the scope switches contexts between the forwards, and the
//      engine's single trace spans them all (a tensor touched by several
//      forwards -- a shared parameter -- is one trace node). Capture
//      each forward's output tensor in a Graph, like the project's (a
//      vector of outputs, exposed by outputs()). Tensors are marked
//      explicitly: Parameter::forward() marks the weights with
//      set_param() when they enter the graph, and graph inputs / compute
//      leaves call set_input().
//   3. Computation(allocator, graph) -- the constructor always
//      allocates: it passes the graph's outputs to allocate(outputs),
//      which plans the trace one output at a time (in graph order), one
//      goal per output: a parameter shared by several outputs is a
//      single decision variable, planned once for all of them -- the
//      first output to plan it decides its split, and that committed
//      split constrains every later output. It then commits the plan:
//      the committed split states
//      are materialized in the MetaDevice's global callback table, and the
//      base allocation places every context's tensors into its own buffers
//      -- the weights are allocated NOW, deferred from load time so the
//      plan could choose their split. Every tensor is allocated with a
//      snapshot of its effective split state.
//      The plan runs once per distinct output set: the same graph
//      computed again (the same outputs) skips the DP, and a changed
//      output set (a different graph) replans.
//   4. A re-plan (a changed device count, a changed communication cost) is
//      another trace + Computation round: forget the round's contexts
//      (their tensors are destroyed with them, so the trace that
//      referenced them is reset), re-trace the fresh round's contexts'
//      forwards through the allocator's engine (the persistent weights
//      are re-traced lazily by set_param()), build the round's Graph, and
//      run Computation: the changed outputs make allocate() replan. The
//      table is replaced for the allocator's tensors, and every context
//      whose snapshots went stale is freed + reallocated; the untouched
//      contexts keep their buffers.
//      An infeasible plan leaves the table in place, so the allocation
//      stays valid for the last good plan.
//   5. Run the real forward() on the ExecutionRuntime: the meta device
//      derives every compute tensor's split state from the callback
//      states and its per-op rules; the planner guarantees that
//      derivation stays in a state the meta backend can execute and that
//      the communication stays minimal.
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
// and the compute cost. The ShardingAllocator's plan round (allocate())
// then runs a dynamic program over the trace, per
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
// distributions cannot be planned; the plan round detects the conflict
// and marks the plan infeasible.
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
//
// The weights live where they act: w_comp and w_mem shape the candidates
// (the ShardingAllocator constructs the ShardingRuntime with them); w_comm
// prices the P -> R bridge. All three are arguments of the
// ShardingAllocator constructor.
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
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ============================================================================
// Mock GGML core (standalone build only; the real headers are not needed)
// ============================================================================

typedef int ggml_type;

struct ggml_tensor {
    char name[64] = {0};   // set by the (mock) weight loader, read by the planner
    ggml_type type = 0;    // needed for ggml_blck_size (axis-0 split alignment)
    int n_dims = 0;
    int64_t ne[4] = {1, 1, 1, 1};   // GGML order, as in the real struct
    bool allocated = false;         // mock bookkeeping (real ggml: data == NULL)
};

struct ggml_context {
    // Owns every tensor created in it (like the real ggml context): the
    // parent engine allocates here, the planner only borrows the pointers.
    std::vector<std::unique_ptr<ggml_tensor>> tensors;

    ggml_tensor* create() {
        auto t = std::make_unique<ggml_tensor>();
        ggml_tensor* raw = t.get();
        tensors.push_back(std::move(t));
        return raw;
    }

    std::vector<ggml_tensor*> all_tensors() const {
        std::vector<ggml_tensor*> v;
        v.reserve(tensors.size());
        for (const auto& t : tensors)
            v.push_back(t.get());
        return v;
    }
};

constexpr ggml_type kMockType = 0;   // the mock graph is dtype-uniform (f32-like, blck size 1)

int ggml_blck_size(ggml_type) { return 1; }   // mock dtypes are f32-like
// Mirror of ggml.h's op enums (the Runtime interface takes them by value).
enum ggml_op_pool {
    GGML_OP_POOL_MAX,
    GGML_OP_POOL_AVG,
    GGML_OP_POOL_COUNT,
};

enum ggml_scale_mode {
    GGML_SCALE_MODE_NEAREST  = 0,
    GGML_SCALE_MODE_BILINEAR = 1,
    GGML_SCALE_MODE_BICUBIC  = 2,
    GGML_SCALE_MODE_LANCZOS3 = 3,
    GGML_SCALE_MODE_LANCZOS4 = 4,
    GGML_SCALE_MODE_COUNT,
};

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
// Mock MetaDevice (mirrors src/ggml/MetaDevice.hpp; the only difference is
// that the underlying devices are mock indices instead of ggml_backend_dev_t)
// ============================================================================

class MetaDevice {
public:
    typedef std::map<const ggml_tensor*, ggml_backend_meta_split_state> Splits;

    explicit MetaDevice(size_t n_devices) : splits_(), n_devices_(n_devices) {}

    size_t count() const { return n_devices_; }

    // The split-state callback table: what the (real)
    // ggml_backend_meta_get_split_state_t callback would return, keyed by
    // the tensor pointer. It is GLOBAL across contexts -- and across every
    // allocator: the ShardingAllocator commits its plan by replacing the
    // entries of the tensors it traced (erase, then insert the new states).
    Splits& splits() { return splits_; }

    // The EFFECTIVE split state of a statically allocated tensor: its
    // planned state if the table has one, otherwise the canonical
    // MIRRORED (nr[0] = 1, n_segments = 1) -- the same default the real
    // callback returns. This is what the allocator queries when it sizes
    // a per-device slice.
    ggml_backend_meta_split_state split(const ggml_tensor* tensor) const {
        const auto it = splits_.find(tensor);
        if (it != splits_.end())
            return it->second;
        ggml_backend_meta_split_state st;
        std::memset(&st, 0, sizeof(st));
        st.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
        st.nr[0] = 1;
        st.n_segments = 1;
        return st;
    }

private:
    Splits splits_;
    size_t n_devices_;
};

// Strict comparison of two split states (the meta backend compares them
// field-wise, see split_states_equal in ggml-backend-meta.cpp). For the
// single-segment states the planner materializes, a strict comparison is
// exactly what the allocator's staleness check wants: any changed
// boundary means a different per-device size.
bool split_state_equal(const ggml_backend_meta_split_state& a, const ggml_backend_meta_split_state& b) {
    if (a.axis != b.axis || a.n_segments != b.n_segments)
        return false;
    for (size_t j = 0; j < sizeof(a.ne) / sizeof(a.ne[0]); ++j)
        if (a.ne[j] != b.ne[j])
            return false;
    for (size_t j = 0; j < sizeof(a.nr) / sizeof(a.nr[0]); ++j)
        if (a.nr[j] != b.nr[j])
            return false;
    return true;
}
// ============================================================================
// Runtime interface (verbatim copy of src/ggml/Runtime.hpp)
// ============================================================================

class Runtime {
public:
    virtual ~Runtime() = default;

    // -------------------------------------------------------------------------
    // Tensor creation / initialization
    // -------------------------------------------------------------------------

    virtual ggml_tensor* new_tensor(
        ggml_type type,
        int n_dims,
        const int64_t* ne
    ) = 0;

    virtual ggml_tensor* new_tensor_1d(
        ggml_type type,
        int64_t ne0
    ) = 0;

    virtual void set_input(
        ggml_tensor* tensor
    ) = 0;

    virtual void set_param(
        ggml_tensor* tensor
    ) = 0;

    virtual ggml_tensor* fill(
        ggml_tensor* tensor,
        float value
    ) = 0;

    // -------------------------------------------------------------------------
    // Copy / cast
    // -------------------------------------------------------------------------

    virtual ggml_tensor* cont(
        ggml_tensor* tensor
    ) = 0;

    virtual ggml_tensor* dup(
        ggml_tensor* tensor
    ) = 0;

    virtual ggml_tensor* cast(
        ggml_tensor* tensor,
        ggml_type type
    ) = 0;

    virtual ggml_tensor* cpy(
        ggml_tensor* src,
        ggml_tensor* dst
    ) = 0;

    // -------------------------------------------------------------------------
    // Unary arithmetic
    // -------------------------------------------------------------------------

    virtual ggml_tensor* sqrt(
        ggml_tensor* tensor
    ) = 0;

    virtual ggml_tensor* exp(
        ggml_tensor* tensor
    ) = 0;

    virtual ggml_tensor* log(
        ggml_tensor* tensor
    ) = 0;

    virtual ggml_tensor* sin(
        ggml_tensor* tensor
    ) = 0;

    virtual ggml_tensor* cos(
        ggml_tensor* tensor
    ) = 0;

    virtual ggml_tensor* sigmoid(
        ggml_tensor* tensor
    ) = 0;

    // -------------------------------------------------------------------------
    // Binary arithmetic
    // -------------------------------------------------------------------------

    virtual ggml_tensor* add(
        ggml_tensor* lhs,
        ggml_tensor* rhs
    ) = 0;

    virtual ggml_tensor* sub(
        ggml_tensor* lhs,
        ggml_tensor* rhs
    ) = 0;

    virtual ggml_tensor* mul(
        ggml_tensor* lhs,
        ggml_tensor* rhs
    ) = 0;

    virtual ggml_tensor* div(
        ggml_tensor* lhs,
        ggml_tensor* rhs
    ) = 0;

    // -------------------------------------------------------------------------
    // Scalar arithmetic
    // -------------------------------------------------------------------------

    virtual ggml_tensor* scale(
        ggml_tensor* tensor,
        float value
    ) = 0;

    virtual ggml_tensor* clamp(
        ggml_tensor* tensor,
        float min,
        float max
    ) = 0;

    // -------------------------------------------------------------------------
    // Matrix operations
    // -------------------------------------------------------------------------

    virtual ggml_tensor* mul_mat(
        ggml_tensor* lhs,
        ggml_tensor* rhs
    ) = 0;

    // -------------------------------------------------------------------------
    // Reshape
    // -------------------------------------------------------------------------

    virtual ggml_tensor* reshape_1d(
        ggml_tensor* tensor,
        int64_t ne0
    ) = 0;

    virtual ggml_tensor* reshape_2d(
        ggml_tensor* tensor,
        int64_t ne0,
        int64_t ne1
    ) = 0;

    virtual ggml_tensor* reshape_3d(
        ggml_tensor* tensor,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2
    ) = 0;

    virtual ggml_tensor* reshape_4d(
        ggml_tensor* tensor,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3
    ) = 0;

    // -------------------------------------------------------------------------
    // Permute / transpose
    // -------------------------------------------------------------------------

    virtual ggml_tensor* permute(
        ggml_tensor* tensor,
        int axis0,
        int axis1,
        int axis2,
        int axis3
    ) = 0;

    // -------------------------------------------------------------------------
    // Views
    // -------------------------------------------------------------------------

    virtual ggml_tensor* view_1d(
        ggml_tensor* tensor,
        int64_t ne0,
        size_t offset
    ) = 0;

    virtual ggml_tensor* view_2d(
        ggml_tensor* tensor,
        int64_t ne0,
        int64_t ne1,
        size_t nb1,
        size_t offset
    ) = 0;

    virtual ggml_tensor* view_3d(
        ggml_tensor* tensor,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        size_t nb1,
        size_t nb2,
        size_t offset
    ) = 0;

    virtual ggml_tensor* view_4d(
        ggml_tensor* tensor,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3,
        size_t nb1,
        size_t nb2,
        size_t nb3,
        size_t offset
    ) = 0;

    // -------------------------------------------------------------------------
    // Repeat / broadcast
    // -------------------------------------------------------------------------

    virtual ggml_tensor* repeat(
        ggml_tensor* tensor,
        ggml_tensor* target
    ) = 0;

    // -------------------------------------------------------------------------
    // Concatenation
    // -------------------------------------------------------------------------

    virtual ggml_tensor* concat(
        ggml_tensor* a,
        ggml_tensor* b,
        int dim
    ) = 0;

    // -------------------------------------------------------------------------
    // Reduction
    // -------------------------------------------------------------------------

    virtual ggml_tensor* sum_rows(
        ggml_tensor* tensor
    ) = 0;

    virtual ggml_tensor * flash_attn_ext(
        ggml_tensor* q,
        ggml_tensor* k,
        ggml_tensor* v,
        ggml_tensor* mask,
        float scale,
        float max_bias,
        float logit_softcap) = 0;

    virtual ggml_tensor * conv_2d_direct(
        ggml_tensor* a,
        ggml_tensor* b,
        int s0,
        int s1,
        int p0,
        int p1,
        int d0,
        int d1) = 0;

    virtual ggml_tensor* get_rows(
        ggml_tensor* a,
        ggml_tensor* b) = 0;

    virtual ggml_tensor* norm(
        ggml_tensor* a,
        float eps) = 0;

    virtual ggml_tensor* rms_norm(
        ggml_tensor* a,
        float eps) = 0;

    virtual ggml_tensor* rope_ext(
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
        float beta_slow) = 0;

    virtual ggml_tensor* pool_2d(
        ggml_tensor* a,
        ggml_op_pool op,
        int k0,
        int k1,
        int s0,
        int s1,
        float p0,
        float p1) = 0;

    virtual ggml_tensor* interpolate(
        ggml_tensor* a,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3,
        uint32_t mode) = 0;

    virtual ggml_tensor* upscale(
        ggml_tensor* a,
        int scale_factor,
        ggml_scale_mode mode) = 0;
};

// ============================================================================
// ShardingRuntime -- the trace of everything
// (drafted in src/ggml/ShardingRuntime.hpp)
//
// Implements the project's Runtime interface: while a module's forward()
// runs through it, every tensor creation is delegated to a parent Runtime
// (which creates the ggml tensors in a ggml_context, exactly as a real
// engine would) and the graph built through it is recorded. ONE engine
// traces everything: every context's forward() runs through the same
// engine (the scope switches contexts), and a tensor touched by several
// forwards (a shared parameter) is ONE trace node. Candidate generation
// is self-contained: the constructor takes the cost weights that shape
// the candidates (w_comp for op compute, w_mem for static storage) and
// the MetaDevice, whose device count sizes the sharded candidates. The
// planning itself -- the DP, the committed splits -- lives in the
// ShardingAllocator that owns it: the allocation is WHY this graph is
// traced.
// ============================================================================

class ShardingRuntime : public Runtime {
public:
    static constexpr int kNoAxis = -1;
    static constexpr double kInf = 1e30;

    struct Dist {
        enum Type { R, S, P };

        Type type = Type::R;
        int axis = kNoAxis;

        bool operator==(const Dist& o) const { return type == o.type && axis == o.axis; }
        bool operator!=(const Dist& o) const { return !(*this == o); }
        bool operator<(const Dist& o) const {
            if (type != o.type) return type < o.type;
            return axis < o.axis;
        }

        std::string to_string() const {
            switch (type) {
                case Type::R: return "R";
                case Type::S: return "S(" + std::to_string(axis) + ")";
                case Type::P: return axis == kNoAxis ? "P" : "P(" + std::to_string(axis) + ")";
            }
            return "?";
        }

        // The plan -> GGML mapping for the axis field of the split state.
        enum ggml_backend_meta_split_axis to_split_axis() const {
            switch (type) {
                case Type::R: return GGML_BACKEND_SPLIT_AXIS_MIRRORED;
                case Type::S: return static_cast<enum ggml_backend_meta_split_axis>(axis);
                case Type::P: return GGML_BACKEND_SPLIT_AXIS_PARTIAL;
            }
            return GGML_BACKEND_SPLIT_AXIS_NONE;
       }

        static Dist replicated() { return Dist{}; }
        static Dist shard(int axis) { return {Type::S, axis}; }
        static Dist partial(int axis = kNoAxis) { return {Type::P, axis}; }
    };

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
        bool is_param = false;                   // set via set_param(): static param, R or S(a) storage decision
        bool is_fixed = false;                   // graph input / compute leaf: externally fixed to R
    };

    // `parent` creates every ggml tensor (in its ggml_context); the engine
    // only borrows the pointers and traces the graph. `device` supplies the
    // device count that sizes the sharded candidates; `w_comp` and `w_mem`
    // are the cost weights the candidates are generated with (see the cost
    // model in the file header).
    ShardingRuntime(Runtime& parent, const MetaDevice& device, double w_comp, double w_mem)
        : parent_(parent), n_devices_(device.count()), w_comp_(w_comp), w_mem_(w_mem) {}

    const std::vector<TraceNode>& nodes() const { return nodes_; }
    const std::vector<ggml_tensor*>& raw_of() const { return raw_of_; }

    // The trace node of a tensor this engine traced (or lazy-traced via
    // set_param / set_input) -- how the allocator registers a goal's root.
    int id_of(ggml_tensor* t) const { return raw_to_id_.at(t); }

    // A round's contexts are going away: their tensors are destroyed
    // with them, so the trace (which references them) is invalid. The
    // next round re-traces through the same engine (the persistent
    // weights are re-traced lazily by set_param()).
    void reset() {
        nodes_.clear();
        raw_of_.clear();
        raw_to_id_.clear();
    }

    // ---------------------------------------------------------------------
    // Runtime: tensor creation / initialization
    // ---------------------------------------------------------------------
    ggml_tensor* new_tensor(ggml_type type, int n_dims, const int64_t* ne) override {
        ggml_tensor* t = parent_.new_tensor(type, n_dims, ne);
        const int rank = std::clamp(n_dims, 0, 4);
        int64_t padded[4] = {1, 1, 1, 1};
        for (int i = 0; i < rank; ++i)
            padded[i] = ne[i];

        // A bare tensor is a static-tensor candidate (R or S(a)); set_param()
        // promotes it to a model param, set_input() fixes it to R.
        trace_op("new_tensor", {}, param_candidates(rank), rank, padded, t);
        return t;
    }

    ggml_tensor* new_tensor_1d(ggml_type type, int64_t ne0) override {
        const int64_t ne[1] = {ne0};
        return new_tensor(type, 1, ne);
    }

    // A tensor whose state is externally fixed: a graph input or a compute
    // leaf (Tensor::empty). The meta backend stores compute leaves in the
    // compute buffer as GGML_OP_NONE, which is MIRRORED -- so the only
    // legal fixed state is R.
    void set_input(ggml_tensor* t) override {
        parent_.set_input(t);
        TraceNode& n = nodes_[ensure_node(t)];
        n.is_fixed = true;
        n.op_name = "input";
        n.is_param = false;
        n.candidates = {{Dist::replicated(), {}, 0.0}};
    }

    // Marks a tensor as a model param (the project's Parameter::forward()
    // calls this when a weight enters the graph): a static tensor whose
    // storage split (R or S(a)) the plan materializes in the callback
    // table, keyed by the tensor name.
    void set_param(ggml_tensor* t) override {
        parent_.set_param(t);
        TraceNode& n = nodes_[ensure_node(t)];
        n.is_param = true;
        n.op_name = "param";
    }

    ggml_tensor* fill(ggml_tensor* t, float value) override {
        // The meta runs FILL through handle_generic (state carries over from
        // the shape template); each device simply fills its own slice.
        ggml_tensor* out = parent_.fill(t, value);
        const int id = get_id(t);
        const int rank = rank_of(t);
        std::vector<Candidate> cands = {{Dist::replicated(), {Dist::replicated()}, 0.0}};
        for (int a = 0; a < rank; ++a)
            cands.push_back({Dist::shard(a), {Dist::shard(a)}, 0.0});
        return traced("fill", {id}, std::move(cands), rank, nodes_[id].ne, out);
    }

    // ---------------------------------------------------------------------
    // Runtime: copy / cast
    // ---------------------------------------------------------------------
    ggml_tensor* cont(ggml_tensor* t) override {
        // GGML_OP_CONT and GGML_OP_RESHAPE share the meta's handle_reshape rule.
        return reinterpret_op("cont", parent_.cont(t), t, nodes_[get_id(t)].ne);
    }

    ggml_tensor* dup(ggml_tensor* t) override {
        // The meta runs DUP with scalar_only: a DUP of a sharded tensor
        // ABORTS. Only a replicated copy is planned.
        ggml_tensor* out = parent_.dup(t);
        const int id = get_id(t);
        return traced("dup", {id}, {{Dist::replicated(), {Dist::replicated()}, w_comp()}}, nodes_[id].rank, nodes_[id].ne, out);
    }

    ggml_tensor* cast(ggml_tensor* t, ggml_type type) override {
        // GGML_OP_CAST is not in the meta backend's switch -> it ABORTs on
        // any split state. The node gets no candidates: any graph that casts
        // is infeasible (keep the graph dtype-uniform instead).
        ggml_tensor* out = parent_.cast(t, type);
        const int id = get_id(t);
        return traced("cast", {id}, {}, nodes_[id].rank, nodes_[id].ne, out);
    }

    ggml_tensor* cpy(ggml_tensor* src, ggml_tensor* dst) override {
        // dst is a shape template; data flows from src. The meta routes a
        // sharded CPY through handle_reshape (the shard remaps to the dst
        // shape); a replicated CPY is a plain copy. handle_reshape requires
        // src_rank <= dst_rank, so sharded candidates only exist then.
        ggml_tensor* out = parent_.cpy(src, dst);
        const int si = get_id(src);
        const int di = get_id(dst);
        const int rank = nodes_[di].rank;
        std::vector<Candidate> cands = {{Dist::replicated(), {Dist::replicated(), Dist::replicated()}, w_comp()}};
        if (nodes_[si].rank <= rank) {
            for (int a = 0; a < nodes_[si].rank; ++a) {
                if (derive_reshape(nodes_[si].ne, nodes_[di].ne, Dist::shard(a)))
                    cands.push_back({Dist::shard(derive_reshape_axis(nodes_[si].ne, nodes_[di].ne, a)), {Dist::shard(a), Dist::replicated()}, sharded_comp()});
            }
        }
        return traced("cpy", {si, di}, std::move(cands), rank, nodes_[di].ne, out);
    }

    // ---------------------------------------------------------------------
    // Runtime: unary arithmetic
    // ---------------------------------------------------------------------
    ggml_tensor* sqrt(ggml_tensor* t) override { return carry_over_op("sqrt", parent_.sqrt(t), t, w_comp()); }
    ggml_tensor* exp(ggml_tensor* t) override { return unsupported_op("exp", parent_.exp(t), t); }
    ggml_tensor* log(ggml_tensor* t) override { return carry_over_op("log", parent_.log(t), t, w_comp()); }
    ggml_tensor* sin(ggml_tensor* t) override { return carry_over_op("sin", parent_.sin(t), t, w_comp()); }
    ggml_tensor* cos(ggml_tensor* t) override { return carry_over_op("cos", parent_.cos(t), t, w_comp()); }
    ggml_tensor* sigmoid(ggml_tensor* t) override { return carry_over_op("sigmoid", parent_.sigmoid(t), t, w_comp()); }

    // ---------------------------------------------------------------------
    // Runtime: binary arithmetic
    // ---------------------------------------------------------------------
    ggml_tensor* add(ggml_tensor* l, ggml_tensor* r) override { return binary_op("add", parent_.add(l, r), l, r); }
    ggml_tensor* sub(ggml_tensor* l, ggml_tensor* r) override { return binary_op("sub", parent_.sub(l, r), l, r); }
    ggml_tensor* mul(ggml_tensor* l, ggml_tensor* r) override { return binary_op("mul", parent_.mul(l, r), l, r); }
    ggml_tensor* div(ggml_tensor* l, ggml_tensor* r) override { return binary_op("div", parent_.div(l, r), l, r); }

    // ---------------------------------------------------------------------
    // Runtime: scalar arithmetic
    // ---------------------------------------------------------------------
    ggml_tensor* scale(ggml_tensor* t, float value) override { return carry_over_op("scale", parent_.scale(t, value), t, w_comp()); }
    ggml_tensor* clamp(ggml_tensor* t, float min, float max) override { return carry_over_op("clamp", parent_.clamp(t, min, max), t, w_comp()); }

    // ---------------------------------------------------------------------
    // Runtime: matrix operations
    // ---------------------------------------------------------------------
    ggml_tensor* mul_mat(ggml_tensor* l, ggml_tensor* r) override {
        // Convention (see nn/Linear): lhs = weight [in, out], rhs = activation.
        // ggml_mul_mat: result ne = {lhs->ne[1], rhs->ne[1], rhs->ne[2], rhs->ne[3]},
        // so the batch dims (and the output rank) come from the rhs.
        ggml_tensor* out = parent_.mul_mat(l, r);
        const int li = get_id(l);
        const int ri = get_id(r);
        const TraceNode& w = nodes_[li];
        const TraceNode& a = nodes_[ri];
        const int64_t out_ne[4] = {w.ne[1], a.ne[1], a.ne[2], a.ne[3]};
        return traced("mul_mat", {li, ri}, mul_mat_candidates(w, a), a.rank, out_ne, out);
    }

    // ---------------------------------------------------------------------
    // Runtime: reshape / permute / views
    // ---------------------------------------------------------------------
    ggml_tensor* reshape_1d(ggml_tensor* t, int64_t ne0) override {
        const int64_t out_ne[4] = {ne0, 1, 1, 1};
        return reinterpret_op("reshape", parent_.reshape_1d(t, ne0), t, out_ne);
    }
    ggml_tensor* reshape_2d(ggml_tensor* t, int64_t ne0, int64_t ne1) override {
        const int64_t out_ne[4] = {ne0, ne1, 1, 1};
        return reinterpret_op("reshape", parent_.reshape_2d(t, ne0, ne1), t, out_ne);
    }
    ggml_tensor* reshape_3d(ggml_tensor* t, int64_t ne0, int64_t ne1, int64_t ne2) override {
        const int64_t out_ne[4] = {ne0, ne1, ne2, 1};
        return reinterpret_op("reshape", parent_.reshape_3d(t, ne0, ne1, ne2), t, out_ne);
    }
    ggml_tensor* reshape_4d(ggml_tensor* t, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) override {
        const int64_t out_ne[4] = {ne0, ne1, ne2, ne3};
        return reinterpret_op("reshape", parent_.reshape_4d(t, ne0, ne1, ne2, ne3), t, out_ne);
    }

    ggml_tensor* permute(ggml_tensor* t, int a0, int a1, int a2, int a3) override {
        ggml_tensor* out = parent_.permute(t, a0, a1, a2, a3);
        const int id = get_id(t);
        const TraceNode& src = nodes_[id];
        const int ax[4] = {a0, a1, a2, a3};
        int64_t out_ne[4] = {1, 1, 1, 1};
        for (int i = 0; i < 4; ++i)
            out_ne[i] = src.ne[ax[i] >= 0 && ax[i] < 4 ? ax[i] : i];

        // GGML_OP_PERMUTE: a shard of the input along axis b reappears on
        // the output axis i with ax[i] == b (the meta's handle_permute).
        std::vector<Candidate> cands = {{Dist::replicated(), {Dist::replicated()}, 0.0}};
        for (int b = 0; b < src.rank; ++b) {
            for (int i = 0; i < src.rank; ++i) {
                if (ax[i] == b)
                    cands.push_back({Dist::shard(i), {Dist::shard(b)}, 0.0});
            }
        }
        return traced("permute", {id}, std::move(cands), src.rank, out_ne, out);
    }

    ggml_tensor* view_1d(ggml_tensor* t, int64_t ne0, size_t offset) override {
        const int64_t out_ne[4] = {ne0, 1, 1, 1};
        return reinterpret_op("view", parent_.view_1d(t, ne0, offset), t, out_ne);
    }
    ggml_tensor* view_2d(ggml_tensor* t, int64_t ne0, int64_t ne1, size_t nb1, size_t offset) override {
        const int64_t out_ne[4] = {ne0, ne1, 1, 1};
        return reinterpret_op("view", parent_.view_2d(t, ne0, ne1, nb1, offset), t, out_ne);
    }
    ggml_tensor* view_3d(ggml_tensor* t, int64_t ne0, int64_t ne1, int64_t ne2, size_t nb1, size_t nb2, size_t offset) override {
        const int64_t out_ne[4] = {ne0, ne1, ne2, 1};
        return reinterpret_op("view", parent_.view_3d(t, ne0, ne1, ne2, nb1, nb2, offset), t, out_ne);
    }
    ggml_tensor* view_4d(ggml_tensor* t, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, size_t nb1, size_t nb2, size_t nb3, size_t offset) override {
        const int64_t out_ne[4] = {ne0, ne1, ne2, ne3};
        return reinterpret_op("view", parent_.view_4d(t, ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset), t, out_ne);
    }

    // ---------------------------------------------------------------------
    // Runtime: repeat / broadcast, concatenation
    // ---------------------------------------------------------------------
    ggml_tensor* repeat(ggml_tensor* t, ggml_tensor* target) override {
        // The meta runs REPEAT through handle_generic: all srcs must be in
        // the SAME state. The target is a compute leaf (fixed R), so only a
        // replicated source can be broadcast.
        ggml_tensor* out = parent_.repeat(t, target);
        const int ti = get_id(t);
        const int ri = get_id(target);
        return traced("repeat", {ti, ri}, {{Dist::replicated(), {Dist::replicated(), Dist::replicated()}, w_comp()}}, nodes_[ri].rank, nodes_[ri].ne, out);
    }

    ggml_tensor* concat(ggml_tensor* a, ggml_tensor* b, int dim) override {
        // GGML_OP_CONCAT (the meta's handle_concat): one operand may be
        // sharded along any axis OTHER than the concat dim; both may be
        // sharded along the same axis. (dim is the GGML axis.)
        ggml_tensor* out = parent_.concat(a, b, dim);
        const int ai = get_id(a);
        const int bi = get_id(b);
        const int rank = std::max(nodes_[ai].rank, nodes_[bi].rank);
        int64_t out_ne[4];
        for (int i = 0; i < 4; ++i)
            out_ne[i] = (i == dim) ? nodes_[ai].ne[i] + nodes_[bi].ne[i] : nodes_[ai].ne[i];

        std::vector<Candidate> cands = {{Dist::replicated(), {Dist::replicated(), Dist::replicated()}, w_comp()}};
        for (int a = 0; a < rank && a != dim; ++a) {
            cands.push_back({Dist::shard(a), {Dist::shard(a), Dist::shard(a)}, sharded_comp()});
            cands.push_back({Dist::shard(a), {Dist::shard(a), Dist::replicated()}, sharded_comp()});
            cands.push_back({Dist::shard(a), {Dist::replicated(), Dist::shard(a)}, sharded_comp()});
        }
        return traced("concat", {ai, bi}, std::move(cands), rank, out_ne, out);
    }

    // ---------------------------------------------------------------------
    // Runtime: reduction
    // ---------------------------------------------------------------------
    ggml_tensor* sum_rows(ggml_tensor* t) override {
        // GGML_OP_SUM_ROWS (the meta's handle_per_row): asserts the src is
        // not sharded along the reduced axis 0 and keeps the src state
        // UNCHANGED (axis preserved, ne preserved).
        ggml_tensor* out = parent_.sum_rows(t);
        const int id = get_id(t);
        const int in_rank = nodes_[id].rank;
        int64_t out_ne[4] = {1, nodes_[id].ne[1], nodes_[id].ne[2], nodes_[id].ne[3]};
        std::vector<Candidate> cands = {{Dist::replicated(), {Dist::replicated()}, w_comp()}};
        for (int a = 1; a < in_rank; ++a)
            cands.push_back({Dist::shard(a), {Dist::shard(a)}, sharded_comp()});
        return traced("sum_rows", {id}, std::move(cands), in_rank > 0 ? in_rank - 1 : 0, out_ne, out);
    }

    // ---------------------------------------------------------------------
    // Runtime: attention
    // ---------------------------------------------------------------------
    ggml_tensor* flash_attn_ext(ggml_tensor* q, ggml_tensor* k, ggml_tensor* v, ggml_tensor* mask, float scale, float max_bias, float logit_softcap) override {
        // GGML_OP_FLASH_ATTN_EXT (the meta's handle_flash_attn_ext): q, k
        // and v are HARD-asserted to be S(2) (the sequence axis of the
        // rank-4 {head_dim, heads, seq, batch} layout), a present mask must
        // be MIRRORED, and the output is S(1). There is no replicated
        // candidate. ggml shape: out ne = {v->ne[0], q->ne[2], q->ne[1], q->ne[3]}.
        ggml_tensor* out = parent_.flash_attn_ext(q, k, v, mask, scale, max_bias, logit_softcap);
        const int qi = get_id(q);
        const int ki = get_id(k);
        const int vi = get_id(v);
        const TraceNode& qt = nodes_[qi];
        const int64_t out_ne[4] = {nodes_[vi].ne[0], qt.ne[2], qt.ne[1], qt.ne[3]};

        std::vector<int> inputs = {qi, ki, vi};
        std::vector<Dist> in_dists = {Dist::shard(2), Dist::shard(2), Dist::shard(2)};
        if (mask) {
            inputs.push_back(get_id(mask));
            in_dists.push_back(Dist::replicated());
        }
        return traced("flash_attn", inputs, {{Dist::shard(1), std::move(in_dists), sharded_comp()}}, qt.rank, out_ne, out);
    }

    // ---------------------------------------------------------------------
    // Runtime: convolution / pooling / resampling (vision)
    // ---------------------------------------------------------------------
    ggml_tensor* conv_2d_direct(ggml_tensor* a, ggml_tensor* b, int s0, int s1, int p0, int p1, int d0, int d1) override {
        // GGML_OP_CONV_2D goes through the meta's handle_generic with
        // scalar_only = true: a sharded src would ABORT. Only the
        // replicated form is planned. a is the kernel [KW, KH, IC, OC], b
        // the input [W, H, C, N]; out = [OW, OH, OC, N].
        ggml_tensor* out = parent_.conv_2d_direct(a, b, s0, s1, p0, p1, d0, d1);
        const int ai = get_id(a);
        const int bi = get_id(b);
        const int64_t out_ne[4] = {
            conv_out_size(nodes_[bi].ne[0], nodes_[ai].ne[0], s0, p0, d0),
            conv_out_size(nodes_[bi].ne[1], nodes_[ai].ne[1], s1, p1, d1),
            nodes_[ai].ne[3],
            nodes_[bi].ne[3],
        };
        return traced("conv_2d", {ai, bi}, {{Dist::replicated(), {Dist::replicated(), Dist::replicated()}, w_comp()}}, 4, out_ne, out);
    }

    ggml_tensor* pool_2d(ggml_tensor* a, ggml_op_pool op, int k0, int k1, int s0, int s1, float p0, float p1) override {
        // GGML_OP_POOL_2D goes through the meta's handle_generic with
        // scalar_only = true: a sharded src would ABORT. Only the
        // replicated form is planned.
        ggml_tensor* out = parent_.pool_2d(a, op, k0, k1, s0, s1, p0, p1);
        const int ai = get_id(a);
        const int64_t out_ne[4] = {
            pool_out_size(nodes_[ai].ne[0], k0, s0, p0),
            pool_out_size(nodes_[ai].ne[1], k1, s1, p1),
            nodes_[ai].ne[2],
            nodes_[ai].ne[3],
        };
        return traced("pool_2d", {ai}, {{Dist::replicated(), {Dist::replicated()}, w_comp()}}, 4, out_ne, out);
    }

    ggml_tensor* interpolate(ggml_tensor* a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, uint32_t mode) override {
        // ggml_interpolate creates a GGML_OP_UPSCALE node, which goes
        // through the meta's handle_generic with scalar_only = true: a
        // sharded src would ABORT. Only the replicated form is planned.
        ggml_tensor* out = parent_.interpolate(a, ne0, ne1, ne2, ne3, mode);
        const int ai = get_id(a);
        const int64_t out_ne[4] = {ne0, ne1, ne2, ne3};
        return traced("interpolate", {ai}, {{Dist::replicated(), {Dist::replicated()}, w_comp()}}, out_rank_of(out_ne), out_ne, out);
    }

    ggml_tensor* upscale(ggml_tensor* a, int scale_factor, ggml_scale_mode mode) override {
        // GGML_OP_UPSCALE goes through the meta's handle_generic with
        // scalar_only = true: a sharded src would ABORT. Only the
        // replicated form is planned. ne0/ne1 are multiplied by the scale
        // factor.
        ggml_tensor* out = parent_.upscale(a, scale_factor, mode);
        const int ai = get_id(a);
        const int64_t out_ne[4] = {nodes_[ai].ne[0] * scale_factor, nodes_[ai].ne[1] * scale_factor, nodes_[ai].ne[2], nodes_[ai].ne[3]};
        return traced("upscale", {ai}, {{Dist::replicated(), {Dist::replicated()}, w_comp()}}, out_rank_of(out_ne), out_ne, out);
    }

    // ---------------------------------------------------------------------
    // Runtime: embeddings / normalization / rotary embeddings
    // ---------------------------------------------------------------------
    ggml_tensor* get_rows(ggml_tensor* a, ggml_tensor* b) override {
        // GGML_OP_GET_ROWS (the meta's handle_get_rows): the data may be
        // sharded along axis 0 while the row indices stay replicated;
        // otherwise everything must be replicated (scalar_only fallback).
        // ggml asserts b = {n_rows, a->ne[2], a->ne[3], 1};
        // out ne = {a->ne[0], b->ne[0], b->ne[1], b->ne[2]}.
        ggml_tensor* out = parent_.get_rows(a, b);
        const int ai = get_id(a);
        const int bi = get_id(b);
        const int64_t out_ne[4] = {nodes_[ai].ne[0], nodes_[bi].ne[0], nodes_[bi].ne[1], nodes_[bi].ne[2]};
        std::vector<Candidate> cands = {
            {Dist::replicated(), {Dist::replicated(), Dist::replicated()}, w_comp()},
            {Dist::shard(0), {Dist::shard(0), Dist::replicated()}, sharded_comp()},
        };
        return traced("get_rows", {ai, bi}, std::move(cands), out_rank_of(out_ne), out_ne, out);
    }

    ggml_tensor* norm(ggml_tensor* a, float eps) override { return per_row_op("norm", parent_.norm(a, eps), a); }
    ggml_tensor* rms_norm(ggml_tensor* a, float eps) override { return per_row_op("rms_norm", parent_.rms_norm(a, eps), a); }

    ggml_tensor* rope_ext(ggml_tensor* a, ggml_tensor* b, ggml_tensor* c,
                          int n_dims, int mode, int n_ctx_orig, float freq_base, float freq_scale,
                          float ext_factor, float attn_factor, float beta_fast, float beta_slow) override {
        // GGML_OP_ROPE (the meta's handle_rope): the position src b must be
        // MIRRORED (hard assert); a's state carries over unchanged, any
        // axis (even 0). c (cos/sin cache) is not state-checked by the
        // meta; a sharded one would fail its ratio asserts, so only a
        // replicated c is planned.
        ggml_tensor* out = parent_.rope_ext(a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        const int ai = get_id(a);
        const TraceNode& at = nodes_[ai];
        const bool has_c = c != nullptr;

        std::vector<int> inputs = {ai, get_id(b)};
        if (has_c)
            inputs.push_back(get_id(c));

        std::vector<Candidate> cands;
        {
            std::vector<Dist> ins = {Dist::replicated(), Dist::replicated()};
            if (has_c) ins.push_back(Dist::replicated());
            cands.push_back({Dist::replicated(), std::move(ins), w_comp()});
        }
        for (int ax = 0; ax < at.rank; ++ax) {
            std::vector<Dist> ins = {Dist::shard(ax), Dist::replicated()};
            if (has_c) ins.push_back(Dist::replicated());
            cands.push_back({Dist::shard(ax), std::move(ins), sharded_comp()});
        }
        return traced("rope", inputs, std::move(cands), at.rank, at.ne, out);
    }

private:
    // ---------------------------------------------------------------------
    // Candidate generation -- exactly the states the meta backend accepts
    // (see the per-op rules in the file header), priced with the cost
    // weights the engine was constructed with. The DP that prices the
    // P -> R bridge with w_comm lives in the ShardingAllocator.
    // ---------------------------------------------------------------------
    std::vector<Candidate> param_candidates(int rank) const;

    // Elementwise unary: the meta carries the src state over unchanged.
    std::vector<Candidate> carry_over_candidates(int rank, double cost) const;

    // ggml binary op: lhs broadcasts rhs against itself (the project's
    // Tensor operators keep the broadcast superset on the left).
    std::vector<Candidate> binary_candidates(const TraceNode& lhs, const TraceNode& rhs, int out_rank) const;

    // mul_mat(lhs = weight [in, out], rhs = activation), result rank = rank(rhs).
    // The meta's handle_mul_mat accepts exactly these four tuples; the
    // row-parallel one additionally GGML_ASSERTs that the weight and
    // activation splits are equal, which holds for near-uniform splits iff
    // the contract dim sizes match.
    std::vector<Candidate> mul_mat_candidates(const TraceNode& w, const TraceNode& a) const;

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
        if (in.type == Dist::Type::R)
            return true;
        if (in.type != Dist::Type::S)
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
    ggml_tensor* reinterpret_op(const char* name, ggml_tensor* out, ggml_tensor* t, const int64_t out_ne[4]) {
        const int id = get_id(t);
        const TraceNode& src = nodes_[id];
        std::vector<Candidate> cands = {{Dist::replicated(), {Dist::replicated()}, 0.0}};
        for (int a = 0; a < src.rank; ++a) {
            if (derive_reshape(src.ne, out_ne, Dist::shard(a)))
                cands.push_back({Dist::shard(derive_reshape_axis(src.ne, out_ne, a)), {Dist::shard(a)}, 0.0});
        }
        return traced(name, {id}, std::move(cands), out_rank_of(out_ne), out_ne, out);
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

    // ggml's output-size formulas (ggml.c), for conv_2d_direct and pool_2d.
    static int64_t conv_out_size(int64_t in, int64_t k, int s, int p, int d) {
        return (in + 2 * p - d * (k - 1) - 1) / s + 1;
    }
    static int64_t pool_out_size(int64_t in, int k, int s, float p) {
        return (in + 2 * p - k) / s + 1;
    }

    ggml_tensor* carry_over_op(const char* name, ggml_tensor* out, ggml_tensor* t, double cost) {
        const int id = get_id(t);
        return traced(name, {id}, carry_over_candidates(nodes_[id].rank, cost), nodes_[id].rank, nodes_[id].ne, out);
    }

    // GGML_OP_NORM / GGML_OP_RMS_NORM (the meta's handle_per_row): the src
    // must not be sharded along the reduced axis 0; the state carries over
    // unchanged (axis preserved, ne preserved).
    ggml_tensor* per_row_op(const char* name, ggml_tensor* out, ggml_tensor* t) {
        const int id = get_id(t);
        const int rank = nodes_[id].rank;
        std::vector<Candidate> cands = {{Dist::replicated(), {Dist::replicated()}, w_comp()}};
        for (int a = 1; a < rank; ++a)
            cands.push_back({Dist::shard(a), {Dist::shard(a)}, sharded_comp()});
        return traced(name, {id}, std::move(cands), rank, nodes_[id].ne, out);
    }

    // An op the meta backend does not handle: zero candidates, so the DP
    // can only report the graph infeasible (with this op named).
    ggml_tensor* unsupported_op(const char* name, ggml_tensor* out, ggml_tensor* t) {
        const int id = get_id(t);
        return traced(name, {id}, {}, nodes_[id].rank, nodes_[id].ne, out);
    }

    // ---------------------------------------------------------------------
    // Tracing helpers
    // ---------------------------------------------------------------------
    int rank_of(ggml_tensor* t) const { return nodes_[get_id(t)].rank; }

    // The trace node for a tensor the planner has not traced yet: it was
    // created outside the plan phase (e.g. by a loader under a different
    // engine), so its shape is read from the ggml tensor and it starts as
    // a bare static-tensor candidate (set_param / set_input refine it).
    int ensure_node(ggml_tensor* t) {
        const auto it = raw_to_id_.find(t);
        if (it != raw_to_id_.end())
            return it->second;
        const int rank = std::clamp(t->n_dims, 0, 4);
        int64_t ne[4] = {1, 1, 1, 1};
        for (int i = 0; i < rank; ++i)
            ne[i] = t->ne[i];
        return trace_op("new_tensor", {}, param_candidates(rank), rank, ne, t);
    }

    int trace_op(const char* name, const std::vector<int>& inputs,
                 std::vector<Candidate> candidates, int rank, const int64_t ne[4], ggml_tensor* out) {
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
        raw_of_.push_back(out);
        raw_to_id_[out] = id;
        return id;
    }

    // trace_op + return the parent-created tensor (the op methods above).
    ggml_tensor* traced(const char* name, const std::vector<int>& inputs,
                        std::vector<Candidate> candidates, int rank, const int64_t ne[4], ggml_tensor* out) {
        trace_op(name, inputs, std::move(candidates), rank, ne, out);
        return out;
    }

    int get_id(ggml_tensor* t) const { return raw_to_id_.at(t); }

    ggml_tensor* binary_op(const char* name, ggml_tensor* out, ggml_tensor* l, ggml_tensor* r) {
        const int li = get_id(l);
        const int ri = get_id(r);
        const int rank = std::max(nodes_[li].rank, nodes_[ri].rank);
        int64_t out_ne[4];
        for (int i = 0; i < 4; ++i)
            out_ne[i] = std::max(nodes_[li].ne[i], nodes_[ri].ne[i]);
        return traced(name, {li, ri}, binary_candidates(nodes_[li], nodes_[ri], rank), rank, out_ne, out);
    }

    // ---------------------------------------------------------------------
    // Cost model (the weights shape the candidates generated above; the
    // DP that prices the bridges with w_comm lives in the ShardingAllocator)
    // ---------------------------------------------------------------------
    double w_comp() const;
    double sharded_comp() const;

    // ---------------------------------------------------------------------
    // State
    // ---------------------------------------------------------------------
    Runtime& parent_;                     // creates the ggml tensors (context)
    size_t n_devices_;                   // from the meta device; sizes the sharded candidates
    double w_comp_;                      // compute of one full (replicated) op
    double w_mem_;                       // per-device storage of a unit tensor

    std::vector<TraceNode> nodes_;
    std::vector<ggml_tensor*> raw_of_;                 // index = trace node id (owned by the parent)
    std::unordered_map<ggml_tensor*, int> raw_to_id_;
};


// ============================================================================
// Mock module framework (mirrors src/ggml/Scope.hpp and src/nn:
// Module / Parameter / Visitor; the Tensor mirrors src/ggml/Tensor.cpp so
// the traced graph structure matches what the real forward() produces)
// ============================================================================

// Mirrors src/ggml/Context.hpp: owns the ggml_context.
class Context {
public:
    Context() : ctx_(new ggml_context()) {}
    ~Context() { delete ctx_; }
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    ggml_context* operator*() const { return ctx_; }

    // Every tensor created in this context, in creation order.
    std::vector<ggml_tensor*> tensors() const { return ctx_->all_tensors(); }

private:
    ggml_context* ctx_;
};

// ============================================================================
// Allocation schema
//
// Mirrors src/ggml/Allocator.hpp: ONE allocator is aware of every context
// registered on it (use(context, device)); allocate() places every
// context's unallocated tensors into per-context buffers over their
// device. On the (mock) meta device that means -- like
// ggml_backend_alloc_ctx_tensors_from_buft on a meta buffer type -- every
// tensor gets a per-device slice sized by its EFFECTIVE split state,
// queried from the device at allocation time.
//
// The allocation is DEFERRED: a context is registered with nothing
// allocated until allocate() runs -- which Computation does only after the
// ShardingAllocator's plan round has committed the split states. That is
// how the weights get their own buffer (allocated separately from every
// compute context) while the plan can still choose their split.
//
// Staleness: the allocator snapshots the effective split state it
// allocated each tensor with. The MetaDevice's table is GLOBAL -- it
// spans every context, and every allocator that commits to it -- so a
// re-plan can change the state of a tensor this allocator already
// allocated (or the device count). The next allocate() then frees the
// context's buffers and realloates it; otherwise only the new tensors are
// placed (a fresh buffer, like ggml_backend_alloc_ctx_tensors_from_buft).
//
// Note for the migration: the real project splits this work in two --
// static tensors go through a context allocator, compute tensors through
// the scheduler's buffers (sized by the meta backend's DERIVED states).
// The mock has no scheduler, so it allocates every context tensor from the
// device's split table (compute tensors simply get the MIRRORED default).
// ============================================================================

constexpr size_t kAlign = 16;   // GGML_MEM_ALIGN: every slice is aligned

// One meta buffer: one simple buffer per underlying device.
class Buffer {
public:
    explicit Buffer(size_t n_devices) : devices_(n_devices) {}

    size_t n_devices() const { return devices_.size(); }
    size_t size(size_t j) const { return devices_[j].size(); }

    // Append `bytes` (already aligned) to device j's buffer; returns the
    // slice's offset.
    size_t append(size_t j, size_t bytes) {
        const size_t off = devices_[j].size();
        devices_[j].resize(off + bytes);
        return off;
    }

private:
    std::vector<std::vector<std::byte>> devices_;
};

// Per-device slice size of a tensor with split state `st` on device `j`
// (single segment, nr == 1, contiguous -- exactly what the planner
// materializes):
//   R / P : every device holds the whole tensor (a PARTIAL result is
//           computed in full on every device, then AllReduced)
//   S(a)  : device j holds st.ne[j] elements along axis a
size_t device_slice_bytes(const ggml_tensor* t, const ggml_backend_meta_split_state& st, size_t j) {
    size_t elems = 1;
    for (int i = 0; i < t->n_dims; ++i)
        elems *= (size_t)t->ne[i];
    const size_t bytes = elems * 4 * (size_t)ggml_blck_size(t->type);   // mock f32
    if (st.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED || st.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL)
        return (bytes + kAlign - 1) / kAlign * kAlign;
    size_t per = 1;
    for (int i = 0; i < 4; ++i)
        if (i != st.axis)
            per *= (size_t)t->ne[i];
    const size_t slice = per * (size_t)st.ne[j] * 4 * (size_t)ggml_blck_size(t->type);
    return (slice + kAlign - 1) / kAlign * kAlign;
}

// The project's Allocator (src/ggml/Allocator.hpp): ONE allocator is aware
// of every context registered on it. The mock omits ggml_backend_buffer_
// usage (the project passes it to every buffer) and queries the device's
// EFFECTIVE split state for every tensor it allocates -- that is what
// makes the staleness snapshot possible (a plain device has no split
// states; nothing ever goes stale there).
class Allocator {
public:
    // One record per allocated tensor: the split state it was allocated
    // with (the staleness snapshot), which buffer holds it, and the
    // per-device slice sizes.
    struct Record {
        ggml_backend_meta_split_state state;
        size_t buffer = 0;
        std::vector<size_t> slice_bytes;
    };

    // Register a context: its unallocated tensors are placed on `device`
    // when allocate() runs.
    void use(Context& context, const MetaDevice& device) {
        Usage usage;
        usage.context = &context;
        usage.device = &device;
        usages_.push_back(std::move(usage));
    }

    // Deregister a context: free its buffers and drop its records. Call
    // before the context is destroyed (its tensors must still be alive).
    void unuse(const Context& context) {
        usages_.erase(std::remove_if(usages_.begin(), usages_.end(),
            [&](const Usage& usage) { return usage.context == &context; }), usages_.end());
    }

    /** @brief Allocates every unallocated tensor in every registered context.
     *
     * @param outputs the graph output tensors (the roots the graph was
     * built to produce). The base allocator ignores them -- it allocates
     * every registered context regardless; the sharded version plans them
     * (one DP goal per output) before allocating, and replans only when
     * the outputs change.
     *
     * Idempotent and staleness-aware: if the device count changed, or the
     * device's effective split state of any ALLOCATED tensor differs from
     * the snapshot taken at its allocation (a re-plan changed a split --
     * including a split being removed, which falls back to MIRRORED), all
     * of the context's buffers are freed and the whole context is
     * reallocated. Otherwise only the new tensors are placed (a fresh
     * buffer, like ggml_backend_alloc_ctx_tensors_from_buft).
     *
     * May be called any number of times; Computation's constructor calls
     * it before running, with the graph's outputs.
     */
    virtual void allocate(const std::vector<ggml_tensor*>& outputs) {
        (void)outputs;
        for (auto& usage : usages_)
            allocate_usage(usage);
    }

    /** @brief Frees every buffer of every registered context and marks all of their tensors unallocated. */
    virtual void reset() {
        for (auto& usage : usages_)
            reset_usage(usage);
    }

    size_t num_contexts() const { return usages_.size(); }
    Context& context(size_t i) const { return *usages_[i].context; }
    const MetaDevice& device(size_t i) const { return *usages_[i].device; }
    size_t num_reallocations(size_t i) const { return usages_[i].reallocations; }
    const std::vector<std::unique_ptr<Buffer>>& buffers(size_t i) const { return usages_[i].buffers; }
    const std::unordered_map<ggml_tensor*, Record>& records(size_t i) const { return usages_[i].records; }

private:
    struct Usage {
        Context* context = nullptr;
        const MetaDevice* device = nullptr;
        std::vector<std::unique_ptr<Buffer>> buffers;
        std::unordered_map<ggml_tensor*, Record> records;
        size_t device_count = 0;
        size_t reallocations = 0;
    };

    void allocate_usage(Usage& usage) {
        if (!usage.records.empty() && stale(usage)) {
            reset_usage(usage);
            ++usage.reallocations;
        }

        int buffer_index = -1;
        for (ggml_tensor* t : usage.context->tensors()) {
            if (t->allocated)
                continue;

            if (buffer_index < 0) {
                buffer_index = (int)usage.buffers.size();
                usage.buffers.push_back(std::make_unique<Buffer>(usage.device->count()));
                usage.device_count = usage.device->count();
            }

            const ggml_backend_meta_split_state st = usage.device->split(t);
            Record rec;
            rec.state = st;
            rec.buffer = (size_t)buffer_index;
            for (size_t j = 0; j < usage.device->count(); ++j)
                rec.slice_bytes.push_back(usage.buffers.back()->append(j, device_slice_bytes(t, st, j)));
            t->allocated = true;
            usage.records[t] = std::move(rec);
        }
    }

    // True when the allocation no longer matches the device's current
    // split states / device count.
    bool stale(const Usage& usage) const {
        if (usage.device->count() != usage.device_count)
            return true;
        for (const auto& [t, rec] : usage.records)
            if (!split_state_equal(usage.device->split(t), rec.state))
                return true;
        return false;
    }

    void reset_usage(Usage& usage) {
        for (ggml_tensor* t : usage.context->tensors())
            t->allocated = false;
        usage.buffers.clear();
        usage.records.clear();
        usage.device_count = 0;
    }

    std::vector<Usage> usages_;
};

// ============================================================================
// ShardingAllocator -- the sharded version of the Allocator: the planning
// state
// (drafted in src/ggml/ShardingAllocator.hpp)
//
// The allocation is WHY we plan: we plan to allocate the tensors optimally
// across the devices. The allocator OWNS the trace: it constructs the ONE
// ShardingRuntime every context's forward() runs through, so every
// context's graph is one subgraph of the single trace.
//
// allocate(outputs) is one allocation round over the graph the outputs
// define:
//
//   1. plan only when the outputs are new -- the first round, or a
//      changed output set (a different graph): the DP runs one goal at a
//      time (in graph order); a tensor shared by several outputs (a
//      parameter consumed by several contexts' forwards) is ONE node,
//      planned exactly once, for all of them -- the first output to plan
//      it decides its split, and that committed split constrains every
//      later output (the meta backend derives one state per tensor).
//      Repeated allocations with the same outputs (the same graph run
//      again) skip the DP entirely;
//   2. commit the plan to the shared resource -- the MetaDevice's split
//      table, GLOBAL across contexts and across every allocator: erase
//      this allocator's parameter entries, insert the materialized states
//      of the plan (the splits are just GGML's way of doing the sharding;
//      the table is what the meta backend queries at runtime);
//   3. run the base allocation over every registered context: every
//      tensor is placed with a snapshot of its current effective split
//      state, and every context whose snapshots went stale (a changed
//      split, a changed device count) is freed + reallocated -- the
//      reallocation the global table makes necessary when the sharding
//      changes.
//
// An infeasible plan leaves the table untouched, so the existing
// allocation stays valid for the last good plan. A round's contexts are
// forgotten (forget) when they go away: their tensors are destroyed with
// them, so the trace that referenced them is reset and the planned
// outputs invalidated -- the next round re-traces and allocate()
// replans from scratch.
// ============================================================================

class ShardingAllocator : public Allocator {
public:

    struct PlanNode {
        int id = 0;
        std::string op_name;
        std::string tensor_name;        // non-empty for params (the callback key)
        ShardingRuntime::Dist produced;
        ShardingRuntime::Dist required;
        std::string bridge;                 // collective between produced and required
        double bridge_cost = 0.0;
    };

    struct Plan {
        double total_cost = 0.0;
        size_t device_count = 0;   // for printing the per-device split sizes
        bool infeasible = false;
        std::string infeasible_reason;
        std::vector<PlanNode> nodes;        // DFS preorder; printed in reverse = execution order
        std::map<int, ShardingRuntime::Dist> callback_dists; // param node id -> storage distribution
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
                    ss << pn.produced.to_string();
                } else {
                    ss << pn.produced.to_string() << " --" << pn.bridge << "--> " << pn.required.to_string();
                }
                ss << "\n";
            }
            return ss.str();
        }
    };

    // `parent` creates every ggml tensor in the contexts; the allocator
    // constructs the ONE ShardingRuntime over it -- the trace every forward
    // runs through (Scope over allocator.runtime()). `device` is the shared
    // resource this allocator commits its plans to: the global split table
    // (plus the device count). `w_comp`/`w_mem` shape the candidates the
    // engine generates; `w_comm` prices the P -> R bridge (see the cost
    // model in the file header).
    ShardingAllocator(Runtime& parent, MetaDevice& device, double w_comp, double w_mem, double w_comm)
        : device_(device), w_comm_(w_comm), runtime_(parent, device, w_comp, w_mem) {}

    // The engine every forward runs through: the trace the allocator plans.
    ShardingRuntime& runtime() { return runtime_; }

    // The communication cost of the next plan round (a re-plan with a
    // changed cost model).
    void set_w_comm(double w_comm) { w_comm_ = w_comm; }

    // `context` is going away: deregister it (free its buffers) and reset
    // the trace -- it spanned the context, so its tensors are destroyed
    // with it (a re-plan re-traces the live contexts, and the shared
    // params -- created in the persistent weights context -- are re-traced
    // lazily by set_param()). The planned outputs go with it. Call before
    // the context is destroyed.
    void forget(const Context& context);

    // One allocation round: plan the graph the outputs define -- but only
    // when the outputs are new (the first round, or a changed output set:
    // a re-plan), one DP goal per output -- commit the plan to the meta
    // device's split table, then (re)allocate every registered context
    // (the base does the buffer work; its staleness snapshots detect the
    // split states this round committed).
    void allocate(const std::vector<ggml_tensor*>& outputs) override;

    // The plan of the last allocate().
    const Plan& plan() const { return last_plan_; }

    // Re-derive every node's split state the way the meta backend does,
    // from the plan's callback table (params) and R (fixed inputs),
    // applying the same per-op rules, and compare against the planned
    // states. Returns false (with an explanation) if the plan is not what
    // the meta device will derive.
    bool verify(const Plan& plan, std::string& error) const;

    // Debug dump of the trace: every node with its shape and the
    // output distributions its candidates can produce.
    std::string dump_trace() const;

    // The committed parameter splits of the last plan round: one entry per
    // shared parameter, spanning every output that consumes it.
    const std::map<const ggml_tensor*, ShardingRuntime::Dist>& decisions() const { return decisions_; }

    const MetaDevice& device() const { return device_; }

private:
    struct Goal {
        int root;
        ShardingRuntime::Dist required;
    };

    struct Bridge {
        std::string name;
        double cost;
    };

    // F(node, d): node produces exactly d.
    struct ExactState {
        bool done = false;
        double cost = ShardingRuntime::kInf;
        int cand = -1;
        std::vector<ShardingRuntime::Dist> in_dists;
    };

    // G(node, d): node satisfies d (produces some d' and bridges d' -> d).
    struct BestState {
        bool done = false;
        double cost = ShardingRuntime::kInf;
        ShardingRuntime::Dist produced;
    };

    // The collective needed to turn a tensor in `from` into the distribution
    // `to`, and its per-device cost. The meta backend's only collective is
    // the AllReduce at a PARTIAL subgraph boundary -- there is no
    // AllGather/ReduceScatter/AllToAll, so everything except P -> R is
    // infeasible. A sharded tensor is consumed sharded through the per-op
    // rules; a full tensor is (re-)produced by a row-parallel mul_mat +
    // the implicit AllReduce.
    Bridge bridge(const ShardingRuntime::Dist& from, const ShardingRuntime::Dist& to) const {
        if (from == to) return {"None", 0.0};
        if (from.type == ShardingRuntime::Dist::Type::P && to.type == ShardingRuntime::Dist::Type::R)
            return {"AllReduce", 0.5 * w_comm_ * comm_factor()};
        return {"Infeasible", ShardingRuntime::kInf};
    }

    double comm_factor() const { return (device_.count() - 1.0) / (double)device_.count(); }


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
    ggml_backend_meta_split_state materialize(const ShardingRuntime::Dist& d, const int64_t ne[4], ggml_type type) const {
        ggml_backend_meta_split_state st;
        std::memset(&st, 0, sizeof(st));
        st.axis = d.to_split_axis();
        st.nr[0] = 1;
        st.n_segments = 1;
        if (d.type == ShardingRuntime::Dist::Type::S) {
            const int64_t gran = d.axis == 0 ? ggml_blck_size(type) : 1;
            int64_t low = 0;
            const int n = (int)device_.count();
            for (int j = 0; j < n; ++j) {
                int64_t high = ne[d.axis] * (int64_t)(j + 1) / n;
                if (j + 1 < n)
                    high = (high / gran) * gran;
                st.ne[j] = high - low;
                low = high;
            }
        }
        return st;
    }

    std::string param_name(int id) const {
        const char* n = runtime_.raw_of()[id]->name;
        return n[0] ? n : ("node" + std::to_string(id));
    }

    // ---------------------------------------------------------------------
    // Dynamic program
    // ---------------------------------------------------------------------
    // Tree DP over the trace, one solve per output (in graph order):
    // F(node, d) pays every shared input once per consumer (a sound
    // bound, used only to select a plan); the plan round
    // recomputes the emitted plan's true per-tensor cost. A tensor shared
    // by several outputs (a param consumed by several contexts' forwards)
    // is ONE node here, so its storage is paid exactly once, for all of
    // them. A param whose split is already committed (by an earlier
    // output's solve) may only produce that split, which keeps the
    // per-output plans consistent: one split per shared param, decided by
    // the first output to plan it.
    //
    // F(node, d): node produces exactly d.
    ExactState& exact(int node, const ShardingRuntime::Dist& d) {
        auto& m = exact_memo_[node][d];
        if (m.done) return m;
        m.done = true;

        const ShardingRuntime::TraceNode& n = runtime_.nodes()[node];
        // A parameter whose split is already committed (planned by an
        // earlier output in this plan round) must keep it: the meta
        // backend derives one state per tensor, shared by every output
        // that consumes it.
        if (n.is_param) {
            const auto c = decisions_.find(runtime_.raw_of()[node]);
            if (c != decisions_.end() && c->second != d)
                return m;   // infeasible state (cost stays kInf)
        }
        for (int c = 0; c < (int)n.candidates.size(); ++c) {
            const ShardingRuntime::Candidate& cand = n.candidates[c];
            if (cand.output != d) continue;

            double cost = cand.comp_cost;
            bool ok = true;
            std::vector<ShardingRuntime::Dist> ins;
            ins.reserve(cand.inputs.size());
            for (size_t i = 0; i < cand.inputs.size(); ++i) {
                const double in_cost = best(n.inputs[i], cand.inputs[i]).cost;
                if (in_cost >= ShardingRuntime::kInf / 2) { ok = false; break; }
                cost += in_cost;
                ins.push_back(cand.inputs[i]);
            }
            if (ok && cost < m.cost)
                m = {true, cost, c, std::move(ins)};
        }
        return m;
    }

    // G(node, d): node satisfies d -- produce some producible d', then bridge.
    BestState& best(int node, const ShardingRuntime::Dist& d) {
        auto& m = best_memo_[node][d];
        if (m.done) return m;
        m.done = true;

        std::set<ShardingRuntime::Dist> producible;
        for (const ShardingRuntime::Candidate& cand : runtime_.nodes()[node].candidates)
            producible.insert(cand.output);

        for (const ShardingRuntime::Dist& p : producible) {
            const double exact_cost = exact(node, p).cost;
            if (exact_cost >= ShardingRuntime::kInf / 2) continue;
            const Bridge b = bridge(p, d);
            if (b.cost >= ShardingRuntime::kInf / 2) continue;
            const double total = exact_cost + b.cost;
            if (total < m.cost)
                m = {true, total, p};
        }
        return m;
    }

    void emit(int node, const ShardingRuntime::Dist& required, Plan& plan, std::set<std::pair<int, ShardingRuntime::Dist>>& emitted) {
        // A tensor consumed several times (even by different outputs) in
        // the same distribution is planned once. A parameter can never be
        // emitted in two distributions (the earlier outputs' commits
        // constrain the DP), but a non-parameter tensor needed in two
        // different distributions would be, and the plan round rejects
        // that: the meta backend derives exactly one state per tensor.
        if (!emitted.insert({node, required}).second) return;

        const BestState& b = best(node, required);
        const Bridge br = bridge(b.produced, required);

        PlanNode pn;
        pn.id = node;
        pn.op_name = runtime_.nodes()[node].op_name;
        pn.tensor_name = runtime_.nodes()[node].is_param ? param_name(node) : "";
        pn.produced = b.produced;
        pn.required = required;
        pn.bridge = std::move(br.name);
        pn.bridge_cost = br.cost;
        plan.nodes.push_back(std::move(pn));

        const ExactState& e = exact(node, b.produced);
        if (e.cand < 0) return;
        const ShardingRuntime::Candidate& cand = runtime_.nodes()[node].candidates[e.cand];
        const ShardingRuntime::TraceNode& n = runtime_.nodes()[node];
        for (size_t i = 0; i < cand.inputs.size(); ++i)
            emit(n.inputs[i], cand.inputs[i], plan, emitted);
    }

    std::string infeasibility_reason(const ShardingRuntime::Dist& required) const {
        std::string r;
        for (const ShardingRuntime::TraceNode& n : runtime_.nodes()) {
            if (n.is_fixed || !n.candidates.empty()) continue;
            if (!r.empty()) r += "; ";
            r += n.op_name + " (node " + std::to_string(n.id) + ") is not supported by the meta backend (no split-state rule)";
        }
        if (r.empty())
            r = "no feasible split plan satisfies the required output distribution " + required.to_string() +
                (decisions_.empty() ? "" : ", given the parameter splits committed by the earlier outputs");
        return r;
    }

    // Solve one plan round over the trace (the DP above + the commit
    // to the meta device's table): one goal per output tensor, in the
    // order the graph provides them. The table is replaced only for a
    // feasible plan: an infeasible one returns early and leaves the
    // previous entries in place, so the allocation stays valid for the
    // last good plan.
    Plan plan_round(const std::vector<ggml_tensor*>& outputs);

    MetaDevice& device_;                 // the shared split-state table + device count
    double w_comm_;

    // The single trace of everything: one ShardingRuntime traced every
    // context's forward (topological order), owned by the allocator.
    ShardingRuntime runtime_;

    // The roots of the last committed plan: allocate() plans only when
    // the outputs differ from these -- a new graph, or a fresh trace
    // after forget(); the same outputs (the same graph computed again)
    // skip the DP.
    std::vector<ggml_tensor*> planned_outputs_;

    std::map<const ggml_tensor*, ShardingRuntime::Dist> decisions_;   // committed param splits, accumulated as the round plans the outputs

    std::map<int, std::map<ShardingRuntime::Dist, ExactState>> exact_memo_;
    std::map<int, std::map<ShardingRuntime::Dist, BestState>> best_memo_;

    Plan last_plan_;
};

void ShardingAllocator::forget(const Context& context) {
    unuse(context);         // free the context's buffers
    runtime_.reset();        // the trace spanned the context; its tensors go away
    planned_outputs_.clear();   // the plan is invalid (a re-plan re-traces the live contexts)
    decisions_.clear();
}

void ShardingAllocator::allocate(const std::vector<ggml_tensor*>& outputs) {
    // Plan the output-tensor graph exactly once: the first round for this
    // output set runs the DP and commits the splits; a changed output set
    // (a different graph) replans; the same outputs (the same graph
    // computed again) skip the DP -- the committed plan already covers
    // them.
    if (!outputs.empty() && !runtime_.nodes().empty() && outputs != planned_outputs_) {
        planned_outputs_ = outputs;
        last_plan_ = plan_round(outputs);
    }
    // The base allocation places every registered context against the
    // device's current split table: the contexts whose snapshots went
    // stale for the split states this round committed are freed and
    // reallocated, the untouched contexts keep their buffers.
    Allocator::allocate(outputs);
}

ShardingAllocator::Plan ShardingAllocator::plan_round(const std::vector<ggml_tensor*>& outputs) {
    decisions_.clear();

    Plan plan;
    if (runtime_.nodes().empty() || outputs.empty())
        return plan;

    plan.device_count = device_.count();
    const std::vector<ShardingRuntime::TraceNode>& nodes = runtime_.nodes();
    const std::vector<ggml_tensor*>& raw = runtime_.raw_of();

    // One goal per output, in graph order: the output's trace node must
    // end in R (the final result is usable on every device). Each
    // output's DP sees the splits committed by the earlier outputs' plans
    // as fixed (the meta backend derives one state per tensor, shared by
    // every output that consumes it), so the first output to plan a
    // parameter decides its split for all of them, and the later outputs
    // adapt to it. The DP is strictly acyclic (topological order), so the
    // memoized recursion terminates.
    std::set<std::pair<int, ShardingRuntime::Dist>> emitted;
    for (ggml_tensor* root : outputs) {
        const Goal g = {runtime_.id_of(root), ShardingRuntime::Dist::replicated()};
        exact_memo_.clear();
        best_memo_.clear();
        const double total = best(g.root, g.required).cost;
        if (total >= ShardingRuntime::kInf / 2) {
            plan.infeasible = true;
            plan.infeasible_reason = infeasibility_reason(g.required);
            return plan;
        }
        const size_t before = plan.nodes.size();
        emit(g.root, g.required, plan, emitted);
        // Commit the splits this output planned for the parameters it uses.
        for (size_t i = before; i < plan.nodes.size(); ++i) {
            const PlanNode& pn = plan.nodes[i];
            if (nodes[pn.id].is_param && decisions_.count(raw[pn.id]) == 0)
                decisions_[raw[pn.id]] = pn.produced;
        }
    }

    for (const PlanNode& pn : plan.nodes)
        if (nodes[pn.id].is_param)
            plan.callback_dists[pn.id] = pn.produced;

    // The meta backend derives exactly one split state per tensor, so a
    // tensor consumed in two different distributions cannot be planned
    // (one static storage layout / one compute layout per tensor) --
    // across outputs as well as within one.
    std::map<int, ShardingRuntime::Dist> single_state;
    for (const PlanNode& pn : plan.nodes) {
        auto [it, inserted] = single_state.insert({pn.id, pn.produced});
        if (!inserted && it->second != pn.produced) {
            plan.infeasible = true;
            plan.infeasible_reason = nodes[pn.id].op_name + " (node " + std::to_string(pn.id) +
                ") is required in both " + it->second.to_string() + " and " +
                pn.produced.to_string() + ", but the meta backend derives a single state per tensor";
            return plan;
        }
    }

    // True cost of the emitted plan: every tensor is planned (and paid
    // for) exactly once, plus its P -> R bridge. The DP above pays a
    // shared input (e.g. the x that both the sigmoid and the mul of
    // x * sigmoid(x) consume) once per consumer, so its total overcounts
    // such subtrees.
    double cost = 0.0;
    for (const PlanNode& pn : plan.nodes) {
        const ExactState& e = exact(pn.id, pn.produced);
        if (e.cand >= 0)
            cost += nodes[pn.id].candidates[e.cand].comp_cost;
        cost += pn.bridge_cost;
    }
    plan.total_cost = cost;

    // Commit: the meta device's split table -- the shared resource, global
    // across contexts and allocators -- is replaced with this allocator's
    // entries for its tensors: erase what this allocator traced, then
    // insert the new states. An infeasible plan (early return above)
    // leaves the previous entries in place, so the allocation stays valid
    // for the last good plan.
    for (const ShardingRuntime::TraceNode& n : nodes)
        if (n.is_param)
            device_.splits().erase(raw[n.id]);

    for (const auto& [id, dist] : plan.callback_dists) {
        decisions_[raw[id]] = dist;
        const ggml_backend_meta_split_state st = materialize(dist, nodes[id].ne, raw[id]->type);
        plan.callback_states[param_name(id)] = st;
        device_.splits()[raw[id]] = st;
    }
    return plan;
}

bool ShardingAllocator::verify(const Plan& plan, std::string& error) const {
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
    std::map<int, ShardingRuntime::Dist> planned;
    for (const PlanNode& pn : plan.nodes)
        planned[pn.id] = pn.produced;

    const std::vector<ShardingRuntime::TraceNode>& nodes = runtime_.nodes();
    std::vector<ShardingRuntime::Dist> visible(nodes.size());
    for (int id = 0; id < (int)nodes.size(); ++id) {
        const ShardingRuntime::TraceNode& n = nodes[id];
        ShardingRuntime::Dist d;
        if (n.is_fixed) {
            d = ShardingRuntime::Dist::replicated();   // compute buffer, GGML_OP_NONE
        } else if (n.is_param) {
            const auto it = plan.callback_dists.find(id);
            if (it == plan.callback_dists.end()) {
                error = param_name(id) + " (node " + std::to_string(id) +
                    ") has no storage state in the plan's callback table";
                return false;
            }
            d = it->second;
        } else {
            std::vector<ShardingRuntime::Dist> in_states;
            in_states.reserve(n.inputs.size());
            for (const int in : n.inputs)
                in_states.push_back(visible[in]);

            const ShardingRuntime::Candidate* match = nullptr;
            int count = 0;
            for (const ShardingRuntime::Candidate& c : n.candidates) {
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
                    ss << (i ? ", " : "") << in_states[i].to_string();
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
                it->second.to_string() + " but the meta backend derives " + d.to_string();
            return false;
        }

        // Consumers of a PARTIAL tensor see MIRRORED: the meta derives
        // source states with assume_sync = true, and the row-parallel
        // mul_mat returns MIRRORED in that mode (the AllReduce happens at
        // the subgraph boundary, before the consumer).
        visible[id] = (d.type == ShardingRuntime::Dist::Type::P) ? ShardingRuntime::Dist::replicated() : d;
    }
    return true;
}

std::string ShardingAllocator::dump_trace() const {
    std::ostringstream ss;
    for (const ShardingRuntime::TraceNode& n : runtime_.nodes()) {
        ss << "  [" << n.id << "] " << n.op_name;
        if (n.is_fixed) ss << " (fixed R)";
        if (n.is_param) ss << " " << (runtime_.raw_of()[n.id]->name[0] ? runtime_.raw_of()[n.id]->name : "?");
        ss << " ne={" << n.ne[0];
        for (int i = 1; i < n.rank; ++i) ss << ", " << n.ne[i];
        ss << "} in={";
        for (size_t i = 0; i < n.inputs.size(); ++i)
            ss << (i ? ", " : "") << n.inputs[i];
        ss << "} candidates:";
        for (const ShardingRuntime::Candidate& c : n.candidates)
            ss << " " << c.output.to_string();
        if (n.candidates.empty())
            ss << " (none -- unsupported by the meta backend)";
        ss << "\n";
    }
    return ss.str();
}

// ShardingRuntime candidate generation: exactly the states the meta backend
// accepts (see the per-op rules in the file header), priced with the
// weights the engine was constructed with.
double ShardingRuntime::w_comp() const { return w_comp_; }
double ShardingRuntime::sharded_comp() const { return w_comp_ / (double)n_devices_; }

std::vector<ShardingRuntime::Candidate> ShardingRuntime::param_candidates(int rank) const {
    std::vector<Candidate> cands;
    cands.push_back({Dist::replicated(), {}, (double)n_devices_ * w_mem_});   // full replica on every device
    for (int a = 0; a < rank; ++a)
        cands.push_back({Dist::shard(a), {}, w_mem_});            // the weight split across devices
    return cands;
}

std::vector<ShardingRuntime::Candidate> ShardingRuntime::carry_over_candidates(int rank, double cost) const {
    std::vector<Candidate> cands;
    cands.push_back({Dist::replicated(), {Dist::replicated()}, cost});
    for (int a = 0; a < rank; ++a)
        cands.push_back({Dist::shard(a), {Dist::shard(a)}, cost / (double)n_devices_});
    return cands;
}

std::vector<ShardingRuntime::Candidate> ShardingRuntime::binary_candidates(const TraceNode& lhs, const TraceNode& rhs, int out_rank) const {
    std::vector<Candidate> cands;
    cands.push_back({Dist::replicated(), {Dist::replicated(), Dist::replicated()}, w_comp_});
    for (int a = 0; a < out_rank && a < lhs.rank; ++a) {
        if (a < rhs.rank)
            cands.push_back({Dist::shard(a), {Dist::shard(a), Dist::shard(a)}, sharded_comp()});
        // The 2nd operand's dim a is size 1: it is a broadcast (the
        // meta's handle_bin_bcast keeps the 1st operand's shard).
        if (rhs.ne[a] == 1)
            cands.push_back({Dist::shard(a), {Dist::shard(a), Dist::replicated()}, sharded_comp()});
    }
    return cands;
}

std::vector<ShardingRuntime::Candidate> ShardingRuntime::mul_mat_candidates(const TraceNode& w, const TraceNode& a) const {
    std::vector<Candidate> cands;
    cands.push_back({Dist::replicated(), {Dist::replicated(), Dist::replicated()}, w_comp_});
    if (w.rank >= 2)
        cands.push_back({Dist::shard(0), {Dist::shard(1), Dist::replicated()}, sharded_comp()});    // column-parallel
    if (a.rank >= 2)
        cands.push_back({Dist::shard(1), {Dist::replicated(), Dist::shard(1)}, sharded_comp()});    // token-parallel
    if (w.rank >= 1 && a.rank >= 1 && w.ne[0] == a.ne[0])
        cands.push_back({Dist::partial(), {Dist::shard(0), Dist::shard(0)}, sharded_comp()}); // row-parallel
    return cands;
}

// Same RAII / static-access pattern as src/ggml/Scope.hpp.
class Scope {
public:
    Scope(Context& context, Runtime& runtime)
        : previous_context_(current_context_), previous_runtime_(current_runtime_)
    {
        current_context_ = &context;
        current_runtime_ = &runtime;
    }
    Scope(const Scope& other)
        : previous_context_(current_context_), previous_runtime_(current_runtime_)
    {
        current_context_ = other.current_context_;
        current_runtime_ = other.current_runtime_;
    }
    ~Scope() {
        current_context_ = previous_context_;
        current_runtime_ = previous_runtime_;
    }

    static Context& context() { return *current_context_; }
    static Runtime& runtime() { return *current_runtime_; }

private:
    inline static thread_local Context* current_context_ = nullptr;
    inline static thread_local Runtime* current_runtime_ = nullptr;
    Context* previous_context_ = nullptr;
    Runtime* previous_runtime_ = nullptr;
};

// Stand-in for src/ggml/ExecutionRuntime: creates every tensor in the
// scoped context (the real one calls the ggml_* constructors there). The
// planner -- and, at runtime, the real ExecutionRuntime -- is driven through
// the Runtime interface; the parent engine owns the ggml tensors.
class ContextRuntime : public Runtime {
public:
    ggml_tensor* new_tensor(ggml_type type, int n_dims, const int64_t* ne) override {
        ggml_tensor* t = (*Scope::context())->create();
        t->type = type;
        t->n_dims = n_dims;
        for (int i = 0; i < n_dims; ++i)
            t->ne[i] = ne[i];
        return t;
    }

    ggml_tensor* new_tensor_1d(ggml_type type, int64_t ne0) override {
        const int64_t ne[1] = {ne0};
        return new_tensor(type, 1, ne);
    }

    void set_input(ggml_tensor*) override {}

    void set_param(ggml_tensor*) override {}

    ggml_tensor* fill(ggml_tensor* t, float) override { return node(t); }

    ggml_tensor* cont(ggml_tensor* t) override { return node(t); }

    ggml_tensor* dup(ggml_tensor* t) override { return node(t); }

    ggml_tensor* cast(ggml_tensor* t, ggml_type type) override {
        ggml_tensor* n = node(t);
        n->type = type;
        return n;
    }

    ggml_tensor* cpy(ggml_tensor* src, ggml_tensor* dst) override { return node(dst); }   // dst is the shape template

    ggml_tensor* sqrt(ggml_tensor* t) override { return node(t); }
    ggml_tensor* exp(ggml_tensor* t) override { return node(t); }
    ggml_tensor* log(ggml_tensor* t) override { return node(t); }
    ggml_tensor* sin(ggml_tensor* t) override { return node(t); }
    ggml_tensor* cos(ggml_tensor* t) override { return node(t); }
    ggml_tensor* sigmoid(ggml_tensor* t) override { return node(t); }

    ggml_tensor* add(ggml_tensor* l, ggml_tensor* r) override { (void)r; return node(l); }
    ggml_tensor* sub(ggml_tensor* l, ggml_tensor* r) override { (void)r; return node(l); }
    ggml_tensor* mul(ggml_tensor* l, ggml_tensor* r) override { (void)r; return node(l); }
    ggml_tensor* div(ggml_tensor* l, ggml_tensor* r) override { (void)r; return node(l); }

    ggml_tensor* scale(ggml_tensor* t, float) override { return node(t); }
    ggml_tensor* clamp(ggml_tensor* t, float, float) override { return node(t); }

    ggml_tensor* mul_mat(ggml_tensor* l, ggml_tensor* r) override {
        ggml_tensor* n = node(r);   // batch dims (and rank) come from the rhs
        n->ne[0] = l->ne[1];
        return n;
    }

    ggml_tensor* reshape_1d(ggml_tensor* t, int64_t ne0) override {
        ggml_tensor* n = node(t);
        n->n_dims = 1;
        n->ne[0] = ne0;
        return n;
    }
    ggml_tensor* reshape_2d(ggml_tensor* t, int64_t ne0, int64_t ne1) override {
        ggml_tensor* n = node(t);
        n->n_dims = 2;
        n->ne[0] = ne0;
        n->ne[1] = ne1;
        return n;
    }
    ggml_tensor* reshape_3d(ggml_tensor* t, int64_t ne0, int64_t ne1, int64_t ne2) override {
        ggml_tensor* n = node(t);
        n->n_dims = 3;
        n->ne[0] = ne0;
        n->ne[1] = ne1;
        n->ne[2] = ne2;
        return n;
    }
    ggml_tensor* reshape_4d(ggml_tensor* t, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) override {
        ggml_tensor* n = node(t);
        n->n_dims = 4;
        n->ne[0] = ne0;
        n->ne[1] = ne1;
        n->ne[2] = ne2;
        n->ne[3] = ne3;
        return n;
    }

    ggml_tensor* permute(ggml_tensor* t, int a0, int a1, int a2, int a3) override {
        ggml_tensor* n = node(t);
        const int ax[4] = {a0, a1, a2, a3};
        for (int i = 0; i < n->n_dims; ++i)
            n->ne[i] = t->ne[ax[i] >= 0 && ax[i] < 4 ? ax[i] : i];
        return n;
    }

    ggml_tensor* view_1d(ggml_tensor* t, int64_t ne0, size_t) override {
        ggml_tensor* n = node(t);
        n->n_dims = 1;
        n->ne[0] = ne0;
        return n;
    }
    ggml_tensor* view_2d(ggml_tensor* t, int64_t ne0, int64_t ne1, size_t, size_t) override {
        ggml_tensor* n = node(t);
        n->n_dims = 2;
        n->ne[0] = ne0;
        n->ne[1] = ne1;
        return n;
    }
    ggml_tensor* view_3d(ggml_tensor* t, int64_t ne0, int64_t ne1, int64_t ne2, size_t, size_t, size_t) override {
        ggml_tensor* n = node(t);
        n->n_dims = 3;
        n->ne[0] = ne0;
        n->ne[1] = ne1;
        n->ne[2] = ne2;
        return n;
    }
    ggml_tensor* view_4d(ggml_tensor* t, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, size_t, size_t, size_t, size_t) override {
        ggml_tensor* n = node(t);
        n->n_dims = 4;
        n->ne[0] = ne0;
        n->ne[1] = ne1;
        n->ne[2] = ne2;
        n->ne[3] = ne3;
        return n;
    }

    ggml_tensor* repeat(ggml_tensor* t, ggml_tensor* target) override { return node(target); }   // target is the shape template

    ggml_tensor* concat(ggml_tensor* a, ggml_tensor* b, int dim) override {
        ggml_tensor* n = node(a);
        n->n_dims = std::max(a->n_dims, b->n_dims);
        for (int i = 0; i < 4; ++i)
            n->ne[i] = (i == dim) ? a->ne[i] + b->ne[i] : a->ne[i];
        return n;
    }

    ggml_tensor* sum_rows(ggml_tensor* t) override {
        ggml_tensor* n = node(t);
        n->n_dims = t->n_dims > 0 ? t->n_dims - 1 : 0;
        n->ne[0] = 1;
        return n;
    }

    ggml_tensor* flash_attn_ext(ggml_tensor* q, ggml_tensor* k, ggml_tensor* v, ggml_tensor* mask, float, float, float) override {
        (void)k; (void)v; (void)mask;
        return node(q);
    }

    ggml_tensor* conv_2d_direct(ggml_tensor* a, ggml_tensor* b, int, int, int, int, int, int) override {
        (void)a;
        return node(b);
    }

    ggml_tensor* get_rows(ggml_tensor* a, ggml_tensor* b) override {
        (void)b;
        return node(a);
    }

    ggml_tensor* norm(ggml_tensor* a, float) override { return node(a); }
    ggml_tensor* rms_norm(ggml_tensor* a, float) override { return node(a); }

    ggml_tensor* rope_ext(ggml_tensor* a, ggml_tensor* b, ggml_tensor* c, int, int, int, float, float, float, float, float, float) override {
        (void)b; (void)c;
        return node(a);
    }

    ggml_tensor* pool_2d(ggml_tensor* a, ggml_op_pool, int, int, int, int, float, float) override { return node(a); }

    ggml_tensor* interpolate(ggml_tensor* a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, uint32_t) override {
        ggml_tensor* n = node(a);
        n->n_dims = 4;
        n->ne[0] = ne0;
        n->ne[1] = ne1;
        n->ne[2] = ne2;
        n->ne[3] = ne3;
        return n;
    }

    ggml_tensor* upscale(ggml_tensor* a, int scale_factor, ggml_scale_mode) override {
        ggml_tensor* n = node(a);
        n->ne[0] *= scale_factor;
        n->ne[1] *= scale_factor;
        return n;
    }

private:
    // Every mock op node: a fresh tensor in the context, dtype and shape
    // copied from the primary source (the shape-changing ops above then
    // fix ne/n_dims). The planner computes the real output shapes
    // analytically; the parent only provides storage.
    ggml_tensor* node(const ggml_tensor* src) {
        ggml_tensor* t = (*Scope::context())->create();
        t->type = src->type;
        t->n_dims = src->n_dims;
        for (int i = 0; i < 4; ++i)
            t->ne[i] = src->ne[i];
        return t;
    }
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
            ? Tensor(Scope::runtime().new_tensor_1d(type, 1), shape, type)
            : Tensor(Scope::runtime().new_tensor(type, (int)shape.rank(), shape.data()), shape, type);
        Scope::runtime().set_input(tensor.t_);
        return tensor;
    }

    ggml_tensor* operator*() const { return t_; }   // the project's Tensor dereferences to ggml_tensor*
    const Shape& shape() const { return shape_; }
    int64_t ndim() const { return shape_.rank(); }
    ggml_type dtype() const { return dtype_; }

    // The mock graph is dtype-uniform: no-op (the project unifies here).
    Tensor to(ggml_type) const { return *this; }

    Tensor contiguous() const {
        return Tensor(Scope::runtime().cont(t_), shape_, dtype_, true);
    }
    Tensor clone() const {
        return Tensor(Scope::runtime().dup(t_), shape_, dtype_, true);
    }
    Tensor scale(float value) const {
        return Tensor(Scope::runtime().scale(t_, value), shape_, dtype_, true);
    }

    // Mirrors the project: ggml_clamp is in-place, so the project clones
    // first. NOTE: that clone (DUP) is not sharding-compatible (the meta
    // aborts on a DUP of a sharded tensor), so sharding-friendly code calls
    // the engine's clamp() directly (see the ReLU demo below).
    Tensor clamp(float a, float b) const {
        auto cloned = clone();
        return Tensor(Scope::runtime().clamp(cloned.t_, a, b), cloned.shape_, dtype_, true);
    }

    Tensor operator+(Tensor rhs) const {
        auto lhs = *this;
        auto target = Shape::broadcast(lhs.shape_, rhs.shape_);

        // ggml_add() natively broadcasts its second argument against the
        // first (the first must already be a broadcast superset). Addition
        // is commutative, so use the argument order that lets ggml broadcast.
        if (Shape::broadcasts(lhs.shape_, rhs.shape_))
            return Tensor(Scope::runtime().add(lhs.t_, rhs.t_), target, dtype_, true);

        if (Shape::broadcasts(rhs.shape_, lhs.shape_))
            return Tensor(Scope::runtime().add(rhs.t_, lhs.t_), target, dtype_, true);

        // Neither shape is a superset of the other: fall back to explicit
        // expansion (the project does the same).
        lhs = lhs.expand(target);
        rhs = rhs.expand(target);
        return Tensor(Scope::runtime().add(lhs.t_, rhs.t_), target, dtype_, true);
    }

    Tensor operator*(Tensor rhs) const {
        auto lhs = *this;
        auto target = Shape::broadcast(lhs.shape_, rhs.shape_);

        if (Shape::broadcasts(lhs.shape_, rhs.shape_))
            return Tensor(Scope::runtime().mul(lhs.t_, rhs.t_), target, dtype_, true);

        if (Shape::broadcasts(rhs.shape_, lhs.shape_))
            return Tensor(Scope::runtime().mul(rhs.t_, lhs.t_), target, dtype_, true);

        lhs = lhs.expand(target);
        rhs = rhs.expand(target);
        return Tensor(Scope::runtime().mul(lhs.t_, rhs.t_), target, dtype_, true);
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
            case 0: return Tensor(Scope::runtime().reshape_1d(src.t_, 1), out, dtype_, true);
            case 1: return Tensor(Scope::runtime().reshape_1d(src.t_, out.data()[0]), out, dtype_, true);
            case 2: return Tensor(Scope::runtime().reshape_2d(src.t_, out.data()[0], out.data()[1]), out, dtype_, true);
            case 3: return Tensor(Scope::runtime().reshape_3d(src.t_, out.data()[0], out.data()[1], out.data()[2]), out, dtype_, true);
            case 4: return Tensor(Scope::runtime().reshape_4d(src.t_, out.data()[0], out.data()[1], out.data()[2], out.data()[3]), out, dtype_, true);
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
        return Tensor(Scope::runtime().repeat(t_, *target), out, dtype_, true);
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
// project's ggml_broadcasts helper): every dim of a is a multiple of the
// matching dim of b (missing dims count as 1).
bool Tensor::Shape::broadcasts(const Shape& a, const Shape& b) {
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

// Mirrors src/ggml/Graph.hpp: the output tensors of the forward(s) the
// computation runs -- a vector of outputs, exposed by outputs(). The real
// Graph also owns the ggml graph and the scheduler; in the PoC the trace
// lives in the ShardingRuntime the forwards ran under (the allocator owns
// it), so the Graph carries just the outputs -- what Computation passes
// to allocate() as the planner's goals (one DP goal per output).
class Graph {
public:
    explicit Graph(std::vector<Tensor> outputs)
        : outputs_(std::move(outputs)) {}

    const std::vector<Tensor>& outputs() const { return outputs_; }

private:
    std::vector<Tensor> outputs_;
};

// Mirrors src/ggml/Computation.hpp: the constructor always allocates the
// buffers the graph needs before computing -- it passes the graph's
// outputs to the allocator, where they become the DP goals. The sharded
// allocator plans the output-tensor graph exactly once (replanning only
// when the outputs change), and the base allocation then (re)allocates
// every registered context (new tensors, stale splits).
class Computation {
public:
    Computation(Allocator& allocator, const Graph& graph)
        : allocator_(&allocator), computed_(false)
    {
        std::vector<ggml_tensor*> outputs;
        for (const Tensor& tensor : graph.outputs())
            outputs.push_back(*tensor);
        allocator_->allocate(outputs);
    }

    Computation& operator ()() {
        // Mock compute: every tensor in every context must be allocated.
        for (size_t i = 0; i < allocator_->num_contexts(); ++i)
            for (ggml_tensor* t : allocator_->context(i).tensors())
                if (!t->allocated)
                    throw std::runtime_error("Computation: a tensor was not allocated");
        computed_ = true;
        return *this;
    }

    ~Computation() {
        assert(computed_ && "Computation was never run. Don't create it if you don't need it.");
    }

    Computation(Computation&) = delete;
    Computation(Computation&&) = delete;
    Computation& operator =(const Computation&) = delete;
    Computation& operator =(Computation&&) = delete;

private:
    Allocator* allocator_;
    bool computed_;
};


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

    // Mirrors the project's Parameter::forward (src/nn/Parameter.hpp): the
    // tensor is created by a loader (visitor) and marked as a param when it
    // enters the graph.
    Tensor forward(Scope scope) {
        scope.runtime().set_param(*tensor_);
        return tensor_;
    }

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
        auto weight = std::static_pointer_cast<Parameter>(modules["weight"])->forward(scope);

        auto y = scope.runtime().mul_mat(*weight, *x);

        if (modules.count("bias")) {
            auto bias = std::static_pointer_cast<Parameter>(modules["bias"])->forward(scope);
            y = scope.runtime().add(y, *bias);
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
        return Tensor(scope.runtime().clamp(*x, 0.0f, std::numeric_limits<float>::infinity()), x.shape());
    }
};

class SiLU : public Module {
public:
    // Mirrors the project's nn/SiLU (src/nn/SiLU.hpp): SiLU(x) = x *
    // sigmoid(x), composed of ops the meta backend supports (MUL +
    // UNARY carry the state over) -- ggml_silu (GGML_OP_SILU) has no
    // split-state rule there.
    Tensor forward(Scope scope, Tensor x) {
        return x * Tensor(scope.runtime().sigmoid(*x), x.shape());
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
        auto& e = scope.runtime();
        const int64_t seq = x.shape()[1];
        const int64_t batch = x.shape()[2];
        const int64_t head_dim = head_dim_;

        auto w_q = std::static_pointer_cast<Parameter>(modules["w_q"])->forward(scope);
        auto w_k = std::static_pointer_cast<Parameter>(modules["w_k"])->forward(scope);
        auto w_v = std::static_pointer_cast<Parameter>(modules["w_v"])->forward(scope);
        auto w_o = std::static_pointer_cast<Parameter>(modules["w_o"])->forward(scope);

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
        auto mask = Tensor::empty(scope.context(), Tensor::Shape{1, 1, 1, batch}, kMockType);

        // ggml: out ne = {v->ne[0], q->ne[2], q->ne[1], q->ne[3]} = {head_dim, heads, seq, batch}
        auto attn = e.flash_attn_ext(qp, kp, vp, mask.t_, 0.5f, 0.0f, 0.0f);

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
        auto& e = scope.runtime();
        auto w1 = std::static_pointer_cast<Parameter>(modules["w1"])->forward(scope);
        auto w2 = std::static_pointer_cast<Parameter>(modules["w2"])->forward(scope);

        auto y = e.mul_mat(*w1, *x);   // ggml_tensor*, {hidden, seq, batch}

        // Positions: a compute leaf (fixed R), GGML {1, seq, 1}.
        auto pos = Tensor::empty(scope.context(), Tensor::Shape{1, x.shape()[1], 1}, kMockType);
        auto r = e.rope_ext(y, pos.t_, nullptr, 0, 0, 0, 10000.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f);

        // Row indices: ggml_get_rows requires b = {n_rows, a->ne[2], a->ne[3], 1};
        // for a = {hidden, seq, batch} that is {n_rows, batch, 1, 1}.
        auto idx = Tensor::empty(scope.context(), Tensor::Shape{x.shape()[0], n_rows_}, kMockType);
        auto g = e.get_rows(r, idx.t_);   // {hidden, n_rows, batch, 1}

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
        auto& e = scope.runtime();
        auto w = std::static_pointer_cast<Parameter>(modules["w_conv"])->forward(scope);
        const int64_t batch = x.shape()[0], h = x.shape()[2], wdt = x.shape()[3];

        // 3x3, pad 1: same spatial size.
        auto c = e.conv_2d_direct(w.t_, x.t_, 1, 1, 1, 1, 1, 1);
        auto n = e.norm(c, 1e-6f);
        auto u = e.upscale(n, 2, GGML_SCALE_MODE_NEAREST);
        auto i = e.interpolate(u, 4 * wdt, 4 * h, out_channels_, batch, GGML_SCALE_MODE_BILINEAR);
        auto p = e.pool_2d(i, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0.0f, 0.0f);
        return Tensor(p, Tensor::Shape{batch, out_channels_, 2 * h, 2 * wdt});
    }

private:
    int64_t out_channels_;
};

// Demo 9 modules: a model with a SHARED trunk and two heads. The trunk's
// weights live in a persistent weights context and are consumed by TWO
// separate compute contexts (one per head) -- the scenario where two
// separately traced contexts refer to the same parameter. The ONE
// ShardingRuntime that traces both contexts keeps the trunk's weights as
// ONE trace node, and the allocator plans them exactly once, for both
// outputs.
class SharedTrunk : public Module {
public:
    SharedTrunk() { modules["fc1"] = std::make_shared<Linear>(8, 16); }

    Tensor forward(Scope scope, Tensor x) {
        return std::static_pointer_cast<Linear>(modules["fc1"])->forward(scope, x);
    }
};

template <class Act>
class Head : public Module {
public:
    explicit Head(const std::string& fc_name)
        : fc_name_(fc_name) {
        modules["act"] = std::make_shared<Act>();
        modules[fc_name_] = std::make_shared<Linear>(16, 8);
    }

    Tensor forward(Scope scope, Tensor x) {
        x = std::static_pointer_cast<Act>(modules["act"])->forward(scope, x);
        return std::static_pointer_cast<Linear>(modules[fc_name_])->forward(scope, x);
    }

private:
    std::string fc_name_;
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
        Tensor t = Tensor(Scope::runtime().new_tensor(kMockType, (int)shape.rank(), shape.data()), shape);
        ggml_set_name(t.t_, name.c_str());
        parameter.set(std::move(t));
    }
};

// ============================================================================
// Demo
// ============================================================================

void print_callback_table(const ShardingAllocator& allocator) {
    const ShardingAllocator::Plan& plan = allocator.plan();
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

std::string split_state_label(const ggml_backend_meta_split_state& st) {
    if (st.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED)
        return "R";
    if (st.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL)
        return "P";
    return "S(" + std::to_string((int)st.axis) + ")";
}

// Human-readable view of what the allocator did to one of its contexts:
// every tensor's snapshot split state and per-device slice sizes, then the
// per-device buffer totals.
std::string print_allocation(const Allocator& allocator, size_t i) {
    std::ostringstream ss;
    ss << "allocation (" << allocator.num_reallocations(i) << " reallocation(s), "
       << allocator.buffers(i).size() << " buffer(s)):\n";
    size_t index = 0;
    for (ggml_tensor* t : allocator.context(i).tensors()) {
        const auto it = allocator.records(i).find(t);
        if (it == allocator.records(i).end())
            continue;
        const Allocator::Record& rec = it->second;
        const char* label = t->name[0] ? t->name : nullptr;
        ss << "  [" << std::setw(3) << index++ << "] "
           << std::left << std::setw(18) << (label ? label : "-") << std::right << " "
           << split_state_label(rec.state);
        if (rec.state.axis >= 0 && rec.state.axis < 4) {
            ss << "  ne=[";
            for (size_t j = 0; j < rec.slice_bytes.size(); ++j)
                ss << (j ? ", " : "") << rec.state.ne[j];
            ss << "]";
        }
        ss << "   slices: ";
        for (size_t j = 0; j < rec.slice_bytes.size(); ++j)
            ss << (j ? " + " : "") << rec.slice_bytes[j] << "B";
        ss << "\n";
    }
    for (size_t b = 0; b < allocator.buffers(i).size(); ++b) {
        const Buffer& buf = *allocator.buffers(i)[b];
        ss << "  buffer #" << b << ": ";
        for (size_t j = 0; j < buf.n_devices(); ++j)
            ss << (j ? " + " : "") << buf.size(j) << "B";
        ss << "\n";
    }
    return ss.str();
}

int main() {
    ggml_time_init();
    ggml_backend_load_all();

    {
        // Demo 1: an MLP that the meta backend can plan. The activation is
        // a ReLU (in-place clamp): the meta backend cannot DUP a sharded
        // tensor, so the sharding-friendly path skips Tensor::clamp's clone.
        MetaDevice meta(2);
        Context context;
        ContextRuntime parent;
        ShardingAllocator allocator(parent, meta, /*w_comp=*/1.0, /*w_mem=*/0.1, /*w_comm=*/0.5);
        allocator.use(context, meta);
        Scope scope(context, allocator.runtime());

        // Graph input: rank-4 activation, PyTorch shape (2, 3, 4, 8)
        // == GGML ne {8, 4, 3, 2}; created like a pipeline input (a
        // compute leaf: fixed R).
        Tensor x = Tensor::empty(context, Tensor::Shape{2, 3, 4, 8}, kMockType);

        MLP<ReLU> model;
        CreateRandomParametersVisitor visitor;
        model.accept(visitor);

        Tensor out = model.forward(scope, x);

        std::cout << "\nDemo 1:\n";
        std::cout << "traced graph:\n" << allocator.dump_trace() << "\n";

        // One plan round: the graph's outputs are the DP goals; the
        // Computation's constructor always allocates -- it plans the
        // output-tensor graph, commits the plan to the meta device's split
        // table, then (re)allocates every registered context.
        Graph graph({out});
        Computation computation(allocator, graph);
        computation();
        const ShardingAllocator::Plan& p = allocator.plan();
        std::cout << p.to_string();

        std::string error;
        if (allocator.verify(p, error))
            std::cout << "verification: OK -- the plan matches the meta backend's split-state derivation\n";
        else
            std::cout << "verification FAILED: " << error << "\n";

        print_callback_table(allocator);

        std::cout << print_allocation(allocator, 0);
    }

    {
        // Demo 2: the same MLP with a SiLU activation, written as
        // x * sigmoid(x) (the project's nn/SiLU does the same): both ops
        // carry the split state over, so the block plans exactly like the
        // clamp version -- and stays sharding-compatible.
        MetaDevice meta(2);
        Context context;
        ContextRuntime parent;
        ShardingAllocator allocator(parent, meta, /*w_comp=*/1.0, /*w_mem=*/0.1, /*w_comm=*/0.5);
        allocator.use(context, meta);
        Scope scope(context, allocator.runtime());
        Tensor x = Tensor::empty(context, Tensor::Shape{2, 3, 4, 8}, kMockType);

        MLP<SiLU> model;
        CreateRandomParametersVisitor visitor;
        model.accept(visitor);

        Tensor out = model.forward(scope, x);

        std::cout << "\nDemo 2:\n";
        std::cout << "traced graph:\n" << allocator.dump_trace() << "\n";
        Graph graph({out});
        Computation computation(allocator, graph);
        computation();
        const ShardingAllocator::Plan& p = allocator.plan();
        std::cout << p.to_string();

        std::string error;
        if (allocator.verify(p, error))
            std::cout << "verification: OK -- the plan matches the meta backend's split-state derivation\n";
        else
            std::cout << "verification FAILED: " << error << "\n";

        print_callback_table(allocator);

        std::cout << print_allocation(allocator, 0);
    }

    {
        // Demo 3: an attention block on flash_attn_ext. The meta backend
        // hard-asserts q, k, v in S(2) and offers no replicated candidate,
        // so with a replicated block input the only feasible plan routes a
        // column-parallel shard onto the sequence axis through zero-cost
        // reshape/permute views (see the module). The block ends with a
        // row-parallel projection: P -> AllReduce -> R.
        MetaDevice meta(2);
        Context context;
        ContextRuntime parent;
        ShardingAllocator allocator(parent, meta, /*w_comp=*/1.0, /*w_mem=*/0.1, /*w_comm=*/0.5);
        allocator.use(context, meta);
        Scope scope(context, allocator.runtime());

        // Block input: PyTorch (batch, seq, hidden) == GGML {hidden, seq, batch}.
        Tensor x = Tensor::empty(context, Tensor::Shape{2, 4, 8}, kMockType);

        AttentionBlock model(/*hidden=*/8, /*heads=*/4);
        CreateRandomParametersVisitor visitor;
        model.accept(visitor);

        Tensor out = model.forward(scope, x);

        std::cout << "\nDemo 3: attention block (flash_attn_ext)\n";
        std::cout << "traced graph:\n" << allocator.dump_trace() << "\n";
        Graph graph({out});
        Computation computation(allocator, graph);
        computation();
        const ShardingAllocator::Plan& p = allocator.plan();
        std::cout << p.to_string();

        std::string error;
        if (allocator.verify(p, error))
            std::cout << "verification: OK -- the plan matches the meta backend's split-state derivation\n";
        else
            std::cout << "verification FAILED: " << error << "\n";

        print_callback_table(allocator);

        std::cout << print_allocation(allocator, 0);
    }

    {
        // Demo 4: rope_ext + get_rows. The shard is created by a
        // column-parallel mul_mat (feature axis S(0)); rope carries the
        // state over (any axis) and get_rows keeps the S(0) shard with
        // replicated indices; the row-parallel projection ends the block
        // (P -> AllReduce -> R).
        MetaDevice meta(2);
        Context context;
        ContextRuntime parent;
        ShardingAllocator allocator(parent, meta, /*w_comp=*/1.0, /*w_mem=*/0.1, /*w_comm=*/0.5);
        allocator.use(context, meta);
        Scope scope(context, allocator.runtime());
        Tensor x = Tensor::empty(context, Tensor::Shape{2, 4, 8}, kMockType);

        RopeGetRows model(/*hidden=*/8, /*n_rows=*/6);
        CreateRandomParametersVisitor visitor;
        model.accept(visitor);

        Tensor out = model.forward(scope, x);

        std::cout << "\nDemo 4: rope_ext + get_rows\n";
        Graph graph({out});
        Computation computation(allocator, graph);
        computation();
        const ShardingAllocator::Plan& p = allocator.plan();
        std::cout << p.to_string();

        std::string error;
        if (allocator.verify(p, error))
            std::cout << "verification: OK -- the plan matches the meta backend's split-state derivation\n";
        else
            std::cout << "verification FAILED: " << error << "\n";

        print_callback_table(allocator);

        std::cout << print_allocation(allocator, 0);
    }

    {
        // Demo 5: a VAE-style upsampler: conv_2d_direct, norm, upscale,
        // interpolate, pool_2d. Every one of these ops is scalar_only in
        // the meta backend, so the whole block must be replicated -- any
        // shard would abort, and the all-replicated plan is the correct
        // (and only) outcome.
        MetaDevice meta(2);
        Context context;
        ContextRuntime parent;
        ShardingAllocator allocator(parent, meta, /*w_comp=*/1.0, /*w_mem=*/0.1, /*w_comm=*/0.5);
        allocator.use(context, meta);
        Scope scope(context, allocator.runtime());

        // Block input: PyTorch (batch, channels, h, w) == GGML {w, h, c, n}.
        Tensor x = Tensor::empty(context, Tensor::Shape{2, 4, 8, 8}, kMockType);

        Upsampler model(/*in_channels=*/4, /*out_channels=*/8);
        CreateRandomParametersVisitor visitor;
        model.accept(visitor);

        Tensor out = model.forward(scope, x);

        std::cout << "\nDemo 5: VAE-style upsampler (conv_2d_direct, upscale, interpolate, pool_2d)\n";
        Graph graph({out});
        Computation computation(allocator, graph);
        computation();
        const ShardingAllocator::Plan& p = allocator.plan();
        std::cout << p.to_string();

        std::string error;
        if (allocator.verify(p, error))
            std::cout << "verification: OK -- the plan matches the meta backend's split-state derivation\n";
        else
            std::cout << "verification FAILED: " << error << "\n";

        print_callback_table(allocator);

        std::cout << print_allocation(allocator, 0);
    }

    {
        // Demo 6: the migration order -- the weights are created by a
        // loader under a plain engine (they live in the context), and the
        // plan phase later runs forward on the allocator's engine,
        // borrowing the existing tensors: set_param() traces them lazily
        // (shape read from the ggml tensor) and set_input() fixes the graph input.
        // The plan must come out identical to demo 1.
        MetaDevice meta(2);
        Context context;
        ContextRuntime parent;
        ShardingAllocator allocator(parent, meta, /*w_comp=*/1.0, /*w_mem=*/0.1, /*w_comm=*/0.5);
        allocator.use(context, meta);
        MLP<ReLU> model;
        {
            Scope loader_scope(context, parent);
            CreateRandomParametersVisitor visitor;
            model.accept(visitor);
        }

        Scope scope(context, allocator.runtime());
        Tensor x = Tensor::empty(context, Tensor::Shape{2, 3, 4, 8}, kMockType);

        Tensor out = model.forward(scope, x);

        std::cout << "\nDemo 6: weights created before the plan phase (loader flow)\n";
        std::cout << "traced graph:\n" << allocator.dump_trace() << "\n";
        Graph graph({out});
        Computation computation(allocator, graph);
        computation();
        const ShardingAllocator::Plan& p = allocator.plan();
        std::cout << p.to_string();

        std::string error;
        if (allocator.verify(p, error))
            std::cout << "verification: OK -- the plan matches the meta backend's split-state derivation\n";
        else
            std::cout << "verification FAILED: " << error << "\n";

        print_callback_table(allocator);

        std::cout << print_allocation(allocator, 0);
    }


    {
        // Demo 7: tensors allocated in MULTIPLE contexts by ONE allocator.
        // Two modules, each with its own context, over ONE meta device:
        // the allocator's ONE engine traces both contexts' forwards
        // (the scope switches contexts; the trace spans them all),
        // the allocator plans that one trace with one goal per output, the
        // device's split table is filled by the single plan, and the base
        // allocation places each context's buffers independently.
        MetaDevice meta(2);

        Context context_a;
        Context context_b;
        ContextRuntime parent;   // stateless: the scope picks the context

        ShardingAllocator allocator(parent, meta, /*w_comp=*/1.0, /*w_mem=*/0.1, /*w_comm=*/0.5);
        allocator.use(context_a, meta);
        allocator.use(context_b, meta);

        // Context A: an MLP (the same graph as demo 1).
        Tensor out_a;
        {
            Scope scope(context_a, allocator.runtime());
            Tensor x = Tensor::empty(context_a, Tensor::Shape{2, 3, 4, 8}, kMockType);
            MLP<ReLU> model_a;
            CreateRandomParametersVisitor visitor;
            model_a.accept(visitor);
            out_a = model_a.forward(scope, x);
        }

        // Context B: the attention block (the same graph as demo 3).
        Tensor out_b;
        {
            Scope scope(context_b, allocator.runtime());
            Tensor x = Tensor::empty(context_b, Tensor::Shape{2, 4, 8}, kMockType);
            AttentionBlock model_b(/*hidden=*/8, /*heads=*/4);
            CreateRandomParametersVisitor visitor;
            model_b.accept(visitor);
            out_b = model_b.forward(scope, x);
        }

        // One goal per output: the single trace is planned once, for both.
        Graph graph({out_a, out_b});

        std::cout << "\nDemo 7: two contexts, one allocator (one trace, one plan + per-context allocation)\n";
        Computation computation(allocator, graph);
        computation();
        const ShardingAllocator::Plan& p = allocator.plan();
        std::cout << "plan (one trace, two outputs):\n" << p.to_string();

        std::string error;
        if (allocator.verify(p, error))
            std::cout << "verification: OK -- the plan matches the meta backend's split-state derivation\n";
        else
            std::cout << "verification FAILED: " << error << "\n";

        // The device's table now holds the plan's 8 params.
        std::cout << "device split table (" << meta.splits().size() << " entries):\n";
        for (const auto& [t, st] : meta.splits())
            std::cout << "  " << std::left << std::setw(18) << (t->name[0] ? t->name : "-") << std::right
                      << " " << split_state_label(st) << "\n";

        std::cout << print_allocation(allocator, 0);
        std::cout << print_allocation(allocator, 1);
    }

    {
        // Demo 8: reallocation when splits change. The weights live in a
        // PERSISTENT loader context (unallocated); the ONE sharded
        // allocator re-plans every round: its engine traces the weights
        // (lazily, by set_param) plus a fresh compute context,
        // with different communication costs: cheap -> the sharded chain
        // with one AllReduce; expensive -> the all-replicated form. The
        // second plan changes the weights' split states in the global
        // split table, so the next allocation detects the stale snapshots
        // and reallocates the weights' buffers. The round's compute
        // context is forgotten afterwards (the round's trace goes with
        // it); the next round re-traces the persistent weights plus a
        // fresh compute context.
        MetaDevice meta(2);

        Context weights;
        ContextRuntime parent;
        MLP<ReLU> model;
        {
            Scope loader_scope(weights, parent);
            CreateRandomParametersVisitor visitor;
            model.accept(visitor);
        }
        ShardingAllocator allocator(parent, meta, /*w_comp=*/1.0, /*w_mem=*/0.1, /*w_comm=*/0.5);
        allocator.use(weights, meta);

        auto run_phase = [&](double w_comm, const char* label) {
            Context compute;
            Tensor out;
            allocator.use(compute, meta);
            allocator.set_w_comm(w_comm);
            {
                Scope scope(compute, allocator.runtime());
                Tensor x = Tensor::empty(compute, Tensor::Shape{2, 3, 4, 8}, kMockType);
                out = model.forward(scope, x);
            }
            Graph graph({out});

            std::cout << "\nDemo 8 " << label << " (w_comm=" << w_comm << "):\n";
            Computation computation(allocator, graph);
            computation();
            std::cout << allocator.plan().to_string();

            std::cout << print_allocation(allocator, 0);
            allocator.forget(compute);   // the round's context goes away (the trace is reset)
        };

        run_phase(/*w_comm=*/0.5, "cheap communication");
        run_phase(/*w_comm=*/10.0, "expensive communication");
    }

    {
        // Demo 9: two compute contexts sharing ONE set of weights -- the
        // scenario the per-context planning state got wrong. The trunk's
        // parameters live in a persistent weights context, and BOTH
        // contexts' forwards set_param() the same tensors through the
        // allocator's ONE engine: the trunk's weights are ONE trace node,
        // planned
        // exactly once, for both outputs (the old scheme planned them once
        // per context, last writer winning in the global split table). The
        // weights are allocated only after the plan (deferred, in their
        // own buffer, separately from each compute context), and a re-plan
        // with expensive communication re-splits them: the split table
        // updates and the next allocation detects the stale snapshots and
        // reallocates the weights.
        MetaDevice meta(2);

        Context weights;
        ContextRuntime parent;
        SharedTrunk trunk;
        Head<ReLU> head_a("fc2a");
        Head<SiLU> head_b("fc2b");
        {
            Scope loader_scope(weights, parent);
            CreateRandomParametersVisitor visitor;
            trunk.accept(visitor);
            head_a.accept(visitor);
            head_b.accept(visitor);
        }
        ShardingAllocator allocator(parent, meta, /*w_comp=*/1.0, /*w_mem=*/0.1, /*w_comm=*/0.5);
        allocator.use(weights, meta);

        auto run_round = [&](double w_comm, const char* label) {
            Context ctx_a;
            Context ctx_b;
            allocator.use(ctx_a, meta);
            allocator.use(ctx_b, meta);
            allocator.set_w_comm(w_comm);

            // The allocator's ONE engine traces both contexts: both
            // forwards set_param() the SAME trunk weight tensors, so the
            // trunk is one node in the trace, shared by both outputs.
            Tensor out_a;
            {
                Scope scope_a(ctx_a, allocator.runtime());
                Tensor x = Tensor::empty(ctx_a, Tensor::Shape{2, 3, 4, 8}, kMockType);
                out_a = head_a.forward(scope_a, trunk.forward(scope_a, x));
            }
            Tensor out_b;
            {
                Scope scope_b(ctx_b, allocator.runtime());
                Tensor x = Tensor::empty(ctx_b, Tensor::Shape{2, 3, 4, 8}, kMockType);
                out_b = head_b.forward(scope_b, trunk.forward(scope_b, x));
            }
            Graph graph({out_a, out_b});

            std::cout << "\nDemo 9 " << label << " (w_comm=" << w_comm << "):\n";
            Computation computation(allocator, graph);
            computation();
            const ShardingAllocator::Plan& p = allocator.plan();
            std::cout << "plan (one trace, two contexts, shared trunk):\n" << p.to_string();

            std::string error;
            if (allocator.verify(p, error))
                std::cout << "verification: OK -- the plan matches the meta backend's split-state derivation\n";
            else
                std::cout << "verification FAILED: " << error << "\n";

            // The allocator carried both outputs: one committed split per
            // parameter, shared by every output that consumes it.
            std::cout << "planning state (committed parameter splits):\n";
            for (const auto& [t, d] : allocator.decisions())
                std::cout << "  " << std::left << std::setw(18) << (t->name[0] ? t->name : "-") << std::right
                          << " " << d.to_string() << "\n";

            // Allocate: the weights get their first allocation NOW
            // (deferred from load time), each context separately. A
            // re-plan that changed a split reallocates the weights.
            std::cout << print_allocation(allocator, 0);
            allocator.forget(ctx_a);   // the round's contexts go away
            allocator.forget(ctx_b);
        };

        run_round(/*w_comm=*/0.5, "cheap communication");
        run_round(/*w_comm=*/10.0, "expensive communication");
    }

}
