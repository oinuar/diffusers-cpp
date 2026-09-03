// ============================================================================
// Tensor-parallel sharding planner -- standalone proof of concept
//
// This file is intentionally self-contained: it mocks the small slice of the
// project's ggml/nn API surface it needs, so it can be built and run
// directly, without CMake:
//
//     g++ -std=c++17 -O2 -o sharding-poc src/main.cpp && ./sharding-poc
//
// The piece that will be migrated into the project is PlannerEngine: it
// implements the exact Engine interface from src/ggml/Engine.hpp and records
// the graph built through it. Everything else (mock ggml types, Context /
// Tensor / Scope / Module framework, main()) is throwaway scaffolding that
// only exists to exercise the planner without pulling in a real backend.
//
// What the planner does
// ---------------------
// Every tensor in the graph is assigned a distribution -- how its data is
// spread over the devices of the parallel group:
//
//   R      replicated   every device holds the whole tensor
//   S(a)   sharded      device i owns the slice along axis a
//   P      partial      device i holds a partial result that must be
//                       AllReduced before it is correct
//
// Axis numbers follow GGML's ne[] order (axis 0 is the fastest dimension)
// and there are at most 4 axes.
//
// While tracing, every op records its "candidates": the distributions it can
// produce natively, the distribution its inputs must be in for that, and the
// compute cost. Candidate axes are generated from the tensor's rank, so
// ranks 1..4 all work. finalize() then runs a dynamic program over
// (node, required distribution):
//
//   exact(n, d) = cheapest way for node n to PRODUCE distribution d
//               = min over candidates c with c.output == d of
//                   c.comp_cost + sum over inputs best(input_i, c.input_i)
//   best(n, d)  = cheapest way for node n to SATISFY distribution d, i.e.
//               produce some d' it can emit and bridge d' -> d with a single
//               collective:
//               = min over producible d' of exact(n, d') + bridge(d', d)
//
// Nodes are traced in topological order (inputs first), so the recursion is
// strictly acyclic and memoization terminates. Bridges:
//
//   P -> R    AllReduce       0.5 * w_comm * (n-1)/n
//   S -> R    AllGather       1.0 * w_comm * (n-1)/n
//   R -> S    local slice     0          (free)
//   P -> S    ReduceScatter   3.0 * w_comm * (n-1)/n
//   S -> S    AllToAll        1.5 * w_comm * (n-1)/n    (different axes)
//   anything else is infeasible
//
// (n-1)/n is the per-device traffic of a collective on a unit-size tensor
// over n devices, and a sharded op computes 1/n of the work, so sharded
// candidates cost w_comp / n.
// ============================================================================

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// ============================================================================
// Mock GGML types (standalone build only; the real headers are not needed)
// ============================================================================

struct ggml_tensor { int dummy; };
struct ggml_context { int dummy; };
typedef int ggml_type;
void ggml_time_init() {}
void ggml_backend_load_all() {}

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
    int rank = 0;                       // 0..4, GGML max dims
    std::vector<int> inputs;            // trace node ids
    std::vector<Candidate> candidates;
    bool is_param = false;
    bool is_input = false;
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
    Dist produced;
    Dist required;
    std::string bridge;                 // collective between produced and required
    double bridge_cost = 0.0;
};

struct Plan {
    double total_cost = 0.0;
    std::vector<PlanNode> nodes;        // DFS preorder; printed in reverse = execution order
    std::map<int, Dist> param_dists;    // param node id -> storage distribution

    std::string to_string() const {
        std::ostringstream ss;
        ss << "=== plan (total cost " << std::fixed << std::setprecision(2) << total_cost << ") ===\n";
        for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
            const PlanNode& pn = *it;
            ss << "  [" << pn.id << "] " << pn.op_name << ": ";
            if (pn.produced == pn.required) {
                ss << dist_to_string(pn.produced);
            } else {
                ss << dist_to_string(pn.produced) << " --" << pn.bridge << "--> " << dist_to_string(pn.required);
            }
            ss << "\n";
        }
        ss << "  params:\n";
        for (const auto& [id, dist] : param_dists)
            ss << "    [" << id << "] -> " << dist_to_string(dist) << "\n";
        ss << "=======================================\n";
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

    // Debug dump of the traced graph: every node with its rank and the
    // output distributions its candidates can produce.
    std::string dump_trace() const;

    // ---------------------------------------------------------------------
    // Engine: tensor creation / initialization
    // ---------------------------------------------------------------------
    ggml_tensor* new_tensor(ggml_context*, ggml_type, int n_dims, const int64_t*) override {
        const int rank = std::clamp(n_dims, 0, 4);
        const int id = trace_op("param", {}, param_candidates(rank), rank);
        nodes_[id].is_param = true;
        return make_tensor(id);
    }

    ggml_tensor* new_tensor_1d(ggml_context* ctx, ggml_type type, int64_t ne0) override {
        const int64_t ne[1] = {ne0};
        return new_tensor(ctx, type, 1, ne);
    }

    void set_input(ggml_tensor* t) override {
        TraceNode& n = nodes_[get_id(t)];
        n.is_input = true;
        n.is_param = false;
        n.candidates = {{rep(), {}, 0.0}};   // a graph input arrives replicated
    }

    ggml_tensor* fill(ggml_tensor* t, float) override {
        // ggml_fill only uses t as a shape template: each device can fill its
        // own slice locally, so every distribution is free and t is not a
        // data dependency.
        const int rank = rank_of(t);
        std::vector<Candidate> cands = {{rep(), {}, 0.0}};
        for (int a = 0; a < rank; ++a) {
            cands.push_back({shard(a), {}, 0.0});
            cands.push_back({partial(a), {}, 0.0});
        }
        return make_tensor(trace_op("fill", {}, std::move(cands), rank));
    }

    // ---------------------------------------------------------------------
    // Engine: copy / cast
    // ---------------------------------------------------------------------
    ggml_tensor* cont(ggml_tensor* t) override { return passthrough_op("cont", t); }
    ggml_tensor* dup(ggml_tensor* t) override { return passthrough_op("dup", t); }
    ggml_tensor* cast(ggml_tensor* t, ggml_type) override { return passthrough_op("cast", t); }

    ggml_tensor* cpy(ggml_tensor* src, ggml_tensor* dst) override {
        // dst is a shape template; data flows from src.
        const int sr = get_id(src);
        const int rank = rank_of(dst);
        std::vector<Candidate> cands = {{rep(), {rep()}, w_comp_}};
        for (int a = 0; a < rank; ++a) {
            cands.push_back({shard(a), {shard(a)}, sharded_comp()});
            cands.push_back({partial(a), {partial(a)}, sharded_comp()});
        }
        return make_tensor(trace_op("cpy", {sr}, std::move(cands), rank));
    }

    // ---------------------------------------------------------------------
    // Engine: unary arithmetic
    // ---------------------------------------------------------------------
    ggml_tensor* neg(ggml_tensor* t) override { return unary_op("neg", t); }
    ggml_tensor* abs(ggml_tensor* t) override { return unary_op("abs", t); }
    ggml_tensor* sqrt(ggml_tensor* t) override { return unary_op("sqrt", t); }
    ggml_tensor* exp(ggml_tensor* t) override { return unary_op("exp", t); }
    ggml_tensor* log(ggml_tensor* t) override { return unary_op("log", t); }
    ggml_tensor* sin(ggml_tensor* t) override { return unary_op("sin", t); }
    ggml_tensor* cos(ggml_tensor* t) override { return unary_op("cos", t); }

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
    ggml_tensor* scale(ggml_tensor* t, float) override { return unary_op("scale", t); }
    ggml_tensor* clamp(ggml_tensor* t, float, float) override { return unary_op("clamp", t); }

    // ---------------------------------------------------------------------
    // Engine: matrix operations
    // ---------------------------------------------------------------------
    ggml_tensor* mul_mat(ggml_tensor* l, ggml_tensor* r) override {
        // Convention (see nn/Linear): lhs = weight [in, out], rhs = activation.
        // ggml_mul_mat: result ne = {lhs->ne[1], rhs->ne[1], rhs->ne[2], rhs->ne[3]},
        // so the batch dims (and the output rank) come from the rhs.
        const int li = get_id(l);
        const int ri = get_id(r);
        return make_tensor(trace_op("mul_mat", {li, ri}, mul_mat_candidates(nodes_[ri].rank), nodes_[ri].rank));
    }

    // ---------------------------------------------------------------------
    // Engine: reshape / permute / views
    // ---------------------------------------------------------------------
    ggml_tensor* reshape_1d(ggml_tensor* t, int64_t) override { return reinterpret_op("reshape", t, 1); }
    ggml_tensor* reshape_2d(ggml_tensor* t, int64_t, int64_t) override { return reinterpret_op("reshape", t, 2); }
    ggml_tensor* reshape_3d(ggml_tensor* t, int64_t, int64_t, int64_t) override { return reinterpret_op("reshape", t, 3); }
    ggml_tensor* reshape_4d(ggml_tensor* t, int64_t, int64_t, int64_t, int64_t) override { return reinterpret_op("reshape", t, 4); }

    ggml_tensor* permute(ggml_tensor* t, int a0, int a1, int a2, int a3) override {
        const int id = get_id(t);
        const int rank = nodes_[id].rank;
        return make_tensor(trace_op("permute", {id}, permute_candidates(rank, a0, a1, a2, a3), rank));
    }

    ggml_tensor* view_1d(ggml_tensor* t, int64_t, size_t) override { return reinterpret_op("view", t, 1); }
    ggml_tensor* view_2d(ggml_tensor* t, int64_t, int64_t, size_t, size_t) override { return reinterpret_op("view", t, 2); }
    ggml_tensor* view_3d(ggml_tensor* t, int64_t, int64_t, int64_t, size_t, size_t, size_t) override { return reinterpret_op("view", t, 3); }
    ggml_tensor* view_4d(ggml_tensor* t, int64_t, int64_t, int64_t, int64_t, size_t, size_t, size_t, size_t) override { return reinterpret_op("view", t, 4); }

    // ---------------------------------------------------------------------
    // Engine: repeat / broadcast, concatenation
    // ---------------------------------------------------------------------
    ggml_tensor* repeat(ggml_tensor* t, ggml_tensor* target) override {
        // The output takes the target's shape; the source is broadcast from a
        // replicated copy (conservative: a sharded source cannot be broadcast
        // to an arbitrary new shape for free).
        const int ti = get_id(t);
        const int rank = rank_of(target);
        std::vector<Candidate> cands = {{rep(), {rep()}, w_comp_}};
        for (int a = 0; a < rank; ++a)
            cands.push_back({shard(a), {rep()}, sharded_comp()});
        return make_tensor(trace_op("repeat", {ti}, std::move(cands), rank));
    }

    ggml_tensor* concat(ggml_tensor* a, ggml_tensor* b, int) override {
        // The concat dim is not modeled: concatenating sharded slices along
        // any axis keeps (or creates) a valid shard, so the elementwise
        // binary candidates apply.
        return binary_op("concat", a, b);
    }

    // ---------------------------------------------------------------------
    // Engine: reduction
    // ---------------------------------------------------------------------
    ggml_tensor* sum_rows(ggml_tensor* t) override {
        const int id = get_id(t);
        const int in_rank = nodes_[id].rank;
        return make_tensor(trace_op("sum_rows", {id}, sum_rows_candidates(in_rank), in_rank - 1));
    }

    ggml_tensor* silu(ggml_tensor* t) override {
        const int id = get_id(t);
        const int rank = nodes_[id].rank;
        std::vector<Candidate> cands = {{rep(), {rep()}, 0.1 * w_comp_}};
        for (int a = 0; a < rank; ++a)
            cands.push_back({shard(a), {shard(a)}, 0.1 * sharded_comp()});
        return make_tensor(trace_op("silu", {id}, std::move(cands), rank));
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
    // `to`, and its per-device cost. Single source of truth for both the DP
    // and the plan output.
    Bridge bridge(const Dist& from, const Dist& to) const {
        if (from == to) return {"None", 0.0};
        const double s = comm_factor();
        if (from.type == DistType::P && to.type == DistType::R) return {"AllReduce", 0.5 * w_comm_ * s};
        if (from.type == DistType::S && to.type == DistType::R) return {"AllGather", 1.0 * w_comm_ * s};
        if (from.type == DistType::R && to.type == DistType::S) return {"Slice", 0.0};
        if (from.type == DistType::P && to.type == DistType::S) return {"ReduceScatter", 3.0 * w_comm_ * s};
        if (from.type == DistType::S && to.type == DistType::S) return {"AllToAll", 1.5 * w_comm_ * s};
        return {"Infeasible", kInf};
    }

    // ---------------------------------------------------------------------
    // Candidate generation (rank-based: axes 0..rank-1, max 4)
    // ---------------------------------------------------------------------
    static Dist axis_or_rep(int axis, int rank) { return axis < rank ? shard(axis) : rep(); }

    std::vector<Candidate> param_candidates(int rank) const {
        std::vector<Candidate> cands;
        cands.push_back({rep(), {}, device_count_ * w_mem_});   // full replica on every device
        for (int a = 0; a < rank; ++a)
            cands.push_back({shard(a), {}, w_mem_});            // the weight split across devices
        return cands;
    }

    // Elementwise unary: the distribution carries over unchanged.
    std::vector<Candidate> unary_candidates(int rank) const {
        std::vector<Candidate> cands;
        cands.push_back({rep(), {rep()}, w_comp_});
        for (int a = 0; a < rank; ++a)
            cands.push_back({shard(a), {shard(a)}, sharded_comp()});
        for (int a = 0; a < rank; ++a)
            cands.push_back({partial(a), {partial(a)}, sharded_comp()});
        return cands;
    }

    // Same layout in and out: distributions carry over, no compute cost.
    std::vector<Candidate> passthrough_candidates(int rank) const {
        std::vector<Candidate> cands;
        cands.push_back({rep(), {rep()}, 0.0});
        for (int a = 0; a < rank; ++a)
            cands.push_back({shard(a), {shard(a)}, 0.0});
        for (int a = 0; a < rank; ++a)
            cands.push_back({partial(a), {partial(a)}, 0.0});
        return cands;
    }

    // Elementwise binary: the output is sharded along axis a when both inputs
    // can be; a lower-rank input (e.g. a bias vector) stays replicated and is
    // sliced locally, which is free.
    std::vector<Candidate> binary_candidates(int out_rank, int lhs_rank, int rhs_rank) const {
        std::vector<Candidate> cands;
        cands.push_back({rep(), {rep(), rep()}, w_comp_});
        for (int a = 0; a < out_rank; ++a)
            cands.push_back({shard(a), {axis_or_rep(a, lhs_rank), axis_or_rep(a, rhs_rank)}, sharded_comp()});
        return cands;
    }

    // mul_mat(lhs = weight [in, out], rhs = activation), result rank = rank(rhs):
    //   result ne = {weight->ne[1], act->ne[1], act->ne[2], act->ne[3]}
    std::vector<Candidate> mul_mat_candidates(int out_rank) const {
        std::vector<Candidate> cands;
        cands.push_back({rep(), {rep(), rep()}, w_comp_});
        cands.push_back({shard(0), {shard(1), rep()}, sharded_comp()});   // column-parallel: split out dim
        cands.push_back({partial(), {shard(0), shard(0)}, sharded_comp()}); // row-parallel: split contract dim -> partial
        if (out_rank > 1)
            cands.push_back({shard(1), {rep(), shard(1)}, sharded_comp()}); // sequence-parallel: split act->ne[1]
        for (int a = 2; a < out_rank; ++a)
            cands.push_back({shard(a), {rep(), shard(a)}, sharded_comp()}); // batch-parallel: split a batch dim
        return cands;
    }

    // ggml_sum_rows reduces ne[0]: input rank r -> output rank r-1.
    std::vector<Candidate> sum_rows_candidates(int in_rank) const {
        std::vector<Candidate> cands;
        cands.push_back({rep(), {rep()}, w_comp_});
        if (in_rank >= 1)
            cands.push_back({partial(), {shard(0)}, sharded_comp()});   // sharded along the reduced axis -> partial
        for (int a = 1; a < in_rank; ++a)
            cands.push_back({shard(a - 1), {shard(a)}, sharded_comp()}); // kept axis shifts down by one
        return cands;
    }

    // Reshape / view re-interpret the same memory. Only a shard along the
    // fastest axis is guaranteed to line up with a single output axis
    // (approximation: assumes the sizes line up); anything else gathers.
    std::vector<Candidate> reinterpret_candidates(int out_rank) const {
        std::vector<Candidate> cands;
        cands.push_back({rep(), {rep()}, 0.0});
        if (out_rank >= 1)
            cands.push_back({shard(0), {shard(0)}, 0.0});
        return cands;
    }

    // ggml_permute(t, ax0..ax3): result ne[i] = t->ne[ax_i], so a shard of
    // the input along axis b reappears on the output axis i with ax_i == b.
    std::vector<Candidate> permute_candidates(int rank, int a0, int a1, int a2, int a3) const {
        const int ax[4] = {a0, a1, a2, a3};
        std::vector<Candidate> cands;
        cands.push_back({rep(), {rep()}, 0.0});
        for (int i = 0; i < rank; ++i) {
            int src = ax[i];
            if (src < 0 || src >= rank) src = i;   // axes beyond the rank are a no-op
            cands.push_back({shard(i), {shard(src)}, 0.0});
        }
        return cands;
    }

    // ---------------------------------------------------------------------
    // Tracing helpers
    // ---------------------------------------------------------------------
    int rank_of(ggml_tensor* t) const { return nodes_[get_id(t)].rank; }

    int trace_op(const char* name, const std::vector<int>& inputs,
                 std::vector<Candidate> candidates, int rank) {
        const int id = (int)nodes_.size();
        TraceNode n;
        n.id = id;
        n.op_name = name;
        n.rank = rank;
        n.inputs = inputs;
        n.candidates = std::move(candidates);
        nodes_.push_back(std::move(n));
        return id;
    }

    ggml_tensor* make_tensor(int id) {
        auto* raw = new ggml_tensor();
        tensors_.emplace_back(raw);
        raw_to_id_[raw] = id;
        return raw;
    }

    int get_id(ggml_tensor* t) const { return raw_to_id_.at(t); }

    ggml_tensor* unary_op(const char* name, ggml_tensor* t) {
        const int id = get_id(t);
        return make_tensor(trace_op(name, {id}, unary_candidates(nodes_[id].rank), nodes_[id].rank));
    }

    ggml_tensor* binary_op(const char* name, ggml_tensor* l, ggml_tensor* r) {
        const int li = get_id(l);
        const int ri = get_id(r);
        const int rank = std::max(nodes_[li].rank, nodes_[ri].rank);
        return make_tensor(trace_op(name, {li, ri}, binary_candidates(rank, nodes_[li].rank, nodes_[ri].rank), rank));
    }

    ggml_tensor* passthrough_op(const char* name, ggml_tensor* t) {
        const int id = get_id(t);
        return make_tensor(trace_op(name, {id}, passthrough_candidates(nodes_[id].rank), nodes_[id].rank));
    }

    ggml_tensor* reinterpret_op(const char* name, ggml_tensor* t, int rank) {
        const int id = get_id(t);
        return make_tensor(trace_op(name, {id}, reinterpret_candidates(rank), rank));
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
        // distribution; merging those into one stored layout is left to the
        // real planner.)
        if (!emitted.insert({node, required}).second) return;

        const BestState& b = best(node, required);
        const Bridge br = bridge(b.produced, required);

        PlanNode pn;
        pn.id = node;
        pn.op_name = nodes_[node].op_name;
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


    // ---------------------------------------------------------------------
    // State
    // ---------------------------------------------------------------------
    int device_count_;
    double w_comm_, w_comp_, w_mem_;

    std::vector<TraceNode> nodes_;
    std::vector<std::unique_ptr<ggml_tensor>> tensors_;
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

    const double total = best(root, required).cost;
    if (total >= kInf / 2) return plan;   // infeasible
    plan.total_cost = total;

    std::set<std::pair<int, Dist>> emitted;
    emit(root, required, plan, emitted);

    for (const PlanNode& pn : plan.nodes)
        if (nodes_[pn.id].is_param)
            plan.param_dists[pn.id] = pn.produced;
    return plan;
}

std::string PlannerEngine::dump_trace() const {
    std::ostringstream ss;
    for (const TraceNode& n : nodes_) {
        ss << "  [" << n.id << "] " << n.op_name;
        if (n.is_input) ss << " (input)";
        if (n.is_param) ss << " (param)";
        ss << " rank=" << n.rank << " in={";
        for (size_t i = 0; i < n.inputs.size(); ++i)
            ss << (i ? ", " : "") << n.inputs[i];
        ss << "} candidates:";
        for (const Candidate& c : n.candidates)
            ss << " " << dist_to_string(c.output);
        ss << "\n";
    }
    return ss.str();
}

// ============================================================================
// Mock module framework (mirrors src/nn: Module / Parameter / Visitor / Scope)
// ============================================================================

class Context {
public:
    ggml_context* ctx_ = nullptr;
    ggml_context* operator*() const { return ctx_; }
};

class Tensor {
public:
    ggml_tensor* t_ = nullptr;
    Tensor() = default;
    explicit Tensor(ggml_tensor* t) : t_(t) {}
    ggml_tensor* operator*() const { return t_; }
    operator bool() const { return t_ != nullptr; }
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

class Parameter;
class Module;

class Visitor {
public:
    virtual void visit(Parameter&, std::vector<std::string> path) {}
    virtual void visit(Module&, std::vector<std::string> path) {}
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
    // The project stores a Tensor::Shape; the planner only needs the rank.
    explicit Parameter(int rank) : rank_(rank) {}

    int rank() const { return rank_; }
    void set(Tensor t) { tensor_ = t; }
    Tensor forward() { return tensor_; }

    void accept(Visitor& visitor, std::vector<std::string> path) override {
        visitor.visit(*this, std::move(path));
    }

private:
    int rank_;
    Tensor tensor_;
};

class Linear : public Module {
public:
    Linear(int64_t, int64_t, bool bias = true) {
        modules["weight"] = std::make_shared<Parameter>(2);   // [out, in] (PyTorch order)
        if (bias)
            modules["bias"] = std::make_shared<Parameter>(1);
    }

    Tensor forward(Scope scope, Tensor x) {
        auto weight = std::static_pointer_cast<Parameter>(modules["weight"])->forward();
        Tensor y(scope.engine().mul_mat(*weight, *x));
        if (modules.count("bias")) {
            auto bias = std::static_pointer_cast<Parameter>(modules["bias"])->forward();
            y = Tensor(scope.engine().add(*y, *bias));
        }
        return y;
    }
};

class SiLU : public Module {
public:
    Tensor forward(Scope scope, Tensor x) {
        return Tensor(scope.engine().silu(*x));
    }
};

class MLP : public Module {
public:
    MLP() {
        modules["fc1"] = std::make_shared<Linear>(8, 16);
        modules["silu"] = std::make_shared<SiLU>();
        modules["fc2"] = std::make_shared<Linear>(16, 8);
    }

    Tensor forward(Scope scope, Tensor x) {
        auto fc1 = std::static_pointer_cast<Linear>(modules["fc1"]);
        auto silu = std::static_pointer_cast<SiLU>(modules["silu"]);
        auto fc2 = std::static_pointer_cast<Linear>(modules["fc2"]);
        x = fc1->forward(scope, x);
        x = silu->forward(scope, x);
        return fc2->forward(scope, x);
    }
};

// Stand-in for the GGUF loader: gives every Parameter a tensor of the right
// rank. (Tests use tiny random models; the planner does not care about values.)
class CreateRandomParametersVisitor : public Visitor {
public:
    void visit(Parameter& parameter, std::vector<std::string>) override {
        const int64_t ne[4] = {1, 1, 1, 1};
        parameter.set(Tensor(Scope::engine().new_tensor(nullptr, 0, parameter.rank(), ne)));
    }
};

// ============================================================================
// Demo
// ============================================================================

int main() {
    ggml_time_init();
    ggml_backend_load_all();

    Context context;
    PlannerEngine planner(/*device_count=*/2, /*w_comm=*/1.0, /*w_comp=*/1.0, /*w_mem=*/0.1);
    Scope scope(context, planner);

    // Graph input: a rank-4 activation, GGML ne {8, 4, 3, 2}.
    const int64_t x_ne[4] = {8, 4, 3, 2};
    ggml_tensor* x_raw = scope.engine().new_tensor(nullptr, 0, 4, x_ne);
    scope.engine().set_input(x_raw);
    Tensor x(x_raw);

    MLP model;
    CreateRandomParametersVisitor visitor;
    model.accept(visitor);

    (void)model.forward(scope, x);

    std::cout << "traced graph:\n" << planner.dump_trace() << "\n";
    std::cout << planner.finalize().to_string();

    return 0;
}
