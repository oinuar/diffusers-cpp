#pragma once

#include "ggml/Runtime.hpp"
#include "ggml/MetaDevice.hpp"
#include <unordered_map>

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
// ShardedAllocator that owns it: the allocation is WHY this graph is
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

    virtual ~ShardingRuntime() = default;

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
    // table, keyed by the tensor name. The param state is re-established
    // here unconditionally: the tensor may have been created through this
    // engine earlier and pinned to R by set_input() (Context::create
    // makes every tensor an input leaf first) -- only set_param's state
    // decides its storage split.
    void set_param(ggml_tensor* t) override {
        parent_.set_param(t);
        TraceNode& n = nodes_[ensure_node(t)];
        n.is_param = true;
        n.is_fixed = false;
        n.op_name = "param";
        n.candidates = param_candidates(n.rank);
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
    ggml_tensor* exp(ggml_tensor* t) override { return carry_over_op("exp", parent_.exp(t), t, w_comp()); }
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
    // P -> R bridge with w_comm lives in the ShardedAllocator.
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
    static int n_dims(const int64_t ne[4]) {
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
        if (axis == n_dims(src_ne) - 1)
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
        if (a == n_dims(src_ne) - 1)
            return n_dims(out_ne) - 1;
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

    static int out_rank_of(const int64_t ne[4]) { return n_dims(ne); }

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
        const int rank = std::clamp(ggml_n_dims(t), 0, 4);
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
    // DP that prices the bridges with w_comm lives in the ShardedAllocator)
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
