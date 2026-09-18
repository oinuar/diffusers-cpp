#pragma once

#include "ggml/Allocator.hpp"
#include "ggml/ShardingRuntime.hpp"
#include <set>
#include <numeric>
#include <sstream>
#include <iomanip>

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
// allocate(contexts, outputs) is one allocation round. The plan is
// incremental: the allocator keeps the plan of every earlier round
// (committed_), so a round plans only what the earlier rounds did not:
//
//   1. the first round plans its outputs (roots): the DP runs one goal
//      at a time (in graph order); a tensor shared by several outputs
//      (a parameter consumed by several contexts' forwards) is ONE node,
//      planned exactly once, for all of them -- the first output to plan
//      it decides its split, and that committed split constrains every
//      later output (the meta backend derives one state per tensor).
//      Every planned tensor is committed, so its split is locked for
//      every later output and every later round;
//   2. a later round (a new output set) solves the DP for its new
//      outputs constrained by the committed plan: a committed tensor
//      may only produce its committed distribution. An output planned
//      by an earlier round is skipped -- the DP is solved for it
//      exactly once, when it first becomes a root (repeated
//      allocations with the same outputs skip the DP entirely). If the
//      DP cannot satisfy a new output given the locked-in splits, the
//      round is infeasible and allocate() throws: the earlier
//      allocation constraints are final, the plan cannot change;
//   3. the round's NEW parameter splits are committed to the shared
//      resource -- the MetaDevice's split table, GLOBAL across contexts
//      and across every allocator (the earlier rounds' entries are
//      untouched; the splits are just GGML's way of doing the sharding;
//      the table is what the meta backend queries at runtime);
//   4. the base allocation runs over the round's contexts, exactly like
//      the base Allocator called again: it allocates only the tensors
//      without a buffer yet, incrementally.
//
// An infeasible round commits nothing (its tentative commits are rolled
// back), so the committed plan -- and the allocation built against it --
// stays valid for the last good plan.
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

                ss << "  [" << pn.id << "] ";

                if (pn.op_name == "param")
                    ss << (pn.tensor_name.empty() ? pn.op_name : pn.tensor_name);
                else
                    ss << pn.op_name << (pn.tensor_name.empty() ? "" : " " + pn.tensor_name);

                ss << ": ";

                if (pn.produced == pn.required) {
                    ss << pn.produced.to_string();
                } else {
                    ss << pn.produced.to_string() << " --" << pn.bridge << "--> " << pn.required.to_string();
                }
                ss << "\n";
            }

            if (!callback_states.empty()) {
                ss << "meta device callback table (ggml_backend_meta_split_state per static tensor):\n";
                for (const auto& [name, st] : callback_states) {
                    ss << "  " << std::left << std::setw(18) << name << std::right;
                    switch (st.axis) {
                        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: ss << "  MIRRORED      ne=[]"; break;
                        case GGML_BACKEND_SPLIT_AXIS_PARTIAL:  ss << "  PARTIAL       ne=[]"; break;
                        default:
                            ss << "  axis " << (int)st.axis << "    ne=[";
                            for (size_t j = 0; j < device_count; ++j)
                                ss << (j ? ", " : "") << st.ne[j];
                            ss << "]";
                    }
                    ss << "\n";
                }
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

    virtual ~ShardingAllocator() = default;

    // The engine every forward runs through: the trace the allocator plans.
    ShardingRuntime& runtime() override { return runtime_; }

    // The communication cost of the next plan round (a re-plan with a
    // changed cost model).
    void set_w_comm(double w_comm) { w_comm_ = w_comm; }

    // One allocation round: plan the outputs incrementally -- the
    // outputs an earlier round did not plan (their DP is constrained by
    // the committed plan: a committed tensor may only produce its
    // committed split; an output already planned is skipped) -- commit
    // the round's new parameter splits to the meta device's split table,
    // then run the base allocation over the round's contexts (it
    // allocates only the tensors without a buffer yet, exactly like the
    // base Allocator called again). A round that cannot be planned
    // against the locked-in splits throws: the previously committed
    // allocation constraints are final.
    void allocate(const std::vector<Context*>& contexts, const std::vector<Tensor>& outputs) override;

    // The plan of the last allocate().
    const Plan& plan() const { return last_plan_; }

    // Debug dump of the trace: every node with its shape and the
    // output distributions its candidates can produce.
    std::string dump_trace() const;

    // The committed parameter splits of every finished round (the
    // public view of committed_): one entry per shared parameter,
    // accumulated over the rounds.
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
        bool feasible = false;   // a candidate exists whose inputs are all feasible
        double cost = ShardingRuntime::kInf;   // min cost over feasible candidates (capped)
        int cand = -1;
        std::vector<ShardingRuntime::Dist> in_dists;
    };

    // G(node, d): node satisfies d (produces some d' and bridges d' -> d).
    struct BestState {
        bool done = false;
        bool feasible = false;   // some producible d' is exact-feasible and bridges to d
        double cost = ShardingRuntime::kInf;   // min cost over feasible productions (capped)
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
    // them. A tensor whose split is already committed (by an earlier
    // output's solve, or by an earlier allocation round) may only produce
    // that split, which keeps the incremental plans consistent: one split
    // per tensor, decided by the first output to plan it.
    //
    // F(node, d): node produces exactly d.
    ExactState& exact(int node, const ShardingRuntime::Dist& d) {
        auto& m = exact_memo_[node][d];
        if (m.done) return m;
        m.done = true;

        // A tensor committed by an earlier round (or an earlier output of
        // this round) is locked to its committed split: it may only
        // produce that distribution, with the candidate and cost it was
        // planned with. Its subgraph is frozen -- the stored state is
        // returned as-is, without re-optimizing its inputs.
        const auto locked = committed_.find(node);
        if (locked != committed_.end()) {
            if (locked->second.produced == d)
                m = locked->second.exact;
            return m;   // otherwise infeasible (cost stays kInf)
        }
        const ShardingRuntime::TraceNode& n = runtime_.nodes()[node];
        for (int c = 0; c < (int)n.candidates.size(); ++c) {
            const ShardingRuntime::Candidate& cand = n.candidates[c];
            if (cand.output != d) continue;

            double cost = cand.comp_cost;
            bool ok = true;
            std::vector<ShardingRuntime::Dist> ins;
            ins.reserve(cand.inputs.size());
            for (size_t i = 0; i < cand.inputs.size(); ++i) {
                const BestState& in = best(n.inputs[i], cand.inputs[i]);
                if (!in.feasible) { ok = false; break; }
                cost += in.cost;
                ins.push_back(cand.inputs[i]);
            }
            if (!ok) continue;
            if (cost > ShardingRuntime::kCostCap)
                cost = ShardingRuntime::kCostCap;   // cap: feasibility is tracked separately
            if (cost < m.cost) {
                m.feasible = true;
                m.cost = cost;
                m.cand = c;
                m.in_dists = std::move(ins);
            }
        }
        return m;
    }

    // G(node, d): node satisfies d -- produce some producible d', then bridge.
    // A goal root (a graph output, see goal_roots_) must be produced
    // exactly in d: the only bridge (P -> R, the AllReduce) is
    // materialized by the meta backend only at a subgraph boundary before
    // a consumer, and its ggml_backend_meta_buffer_get_tensor (assume_sync
    // = false) has no PARTIAL case, so a PARTIAL output can never be read.
    BestState& best(int node, const ShardingRuntime::Dist& d) {
        auto& m = best_memo_[node][d];
        if (m.done) return m;
        m.done = true;

        const bool exact_only = goal_roots_.count(node) != 0;

        // A committed tensor is producible only in its committed
        // distribution; an uncommitted one in every candidate output.
        std::set<ShardingRuntime::Dist> producible;
        const auto locked = committed_.find(node);
        if (locked != committed_.end())
            producible.insert(locked->second.produced);
        else
            for (const ShardingRuntime::Candidate& cand : runtime_.nodes()[node].candidates)
                producible.insert(cand.output);

        for (const ShardingRuntime::Dist& p : producible) {
            if (exact_only && p != d) continue;
            const ExactState& e = exact(node, p);
            if (!e.feasible) continue;
            const Bridge b = bridge(p, d);
            if (b.cost >= ShardingRuntime::kInf / 2) continue;
            double total = e.cost + b.cost;
            if (total > ShardingRuntime::kCostCap)
                total = ShardingRuntime::kCostCap;   // cap: feasibility is tracked separately
            if (total < m.cost) {
                m.feasible = true;
                m.cost = total;
                m.produced = p;
            }
        }
        return m;
    }

    void emit(int node, const ShardingRuntime::Dist& required, Plan& plan, std::set<std::pair<int, ShardingRuntime::Dist>>& emitted) {
        // A committed tensor keeps the split the round that planned it
        // decided (locked in committed_): it is already in the
        // cumulative plan, so skip it -- this round's plan covers only
        // the tensors it plans.
        if (committed_.count(node))
            return;
        //
        // A tensor consumed several times (even by different outputs) in
        // the same distribution is planned once. A tensor can never be
        // committed in two distributions (the earlier commits constrain
        // the DP), but one needed in two different distributions within
        // this round would be, and the plan round rejects that: the meta
        // backend derives exactly one state per tensor.
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
        std::vector<std::string> r;
        for (const ShardingRuntime::TraceNode& n : runtime_.nodes()) {
            if (n.is_fixed)
                continue;

            // If there are no candidates for node, the plan is trivially infeasible.
            if (n.candidates.empty())
                r.push_back(n.op_name + " (node " + std::to_string(n.id) + ") is not supported by the meta backend (no split-state rule)");

            // flash_attn_ext is only splittable with q/k/v sharded along the
            // sequence axis and the output sharded along the heads axis (S(1)).
            // That committed S(1) split must be uniform across the devices,
            // so the heads dim (axis 1 of the flash_attn output) must divide by
            // the device count.
            else if (n.op_name == "flash_attn" && n.ne[1] % device_.count() != 0)
                r.push_back(n.op_name + " (node " + std::to_string(n.id) + ") " +
                    "must be sharded along the heads axis (S(1)), but its size " +
                    std::to_string(n.ne[1]) + " is not divisible by the device count " +
                    std::to_string(device_.count()));
        }

        // Infeasibility is because of unknown reason.
        if (r.empty())
            r.push_back("no feasible split plan satisfies the required output distribution " + required.to_string() +
                (committed_.empty() ? "" :
                 ", given the splits locked in by the earlier allocation rounds: the previously committed outputs constrain the plan"));

        return std::accumulate(std::begin(r), std::end(r), std::string(),
            [](const std::string& acc, const std::string& x) {
                if (acc.empty())
                    return x;

                return acc + "\n" + x;
            }
        );
    }

    // Solve one allocation round incrementally: plan the outputs that
    // are not committed yet (one DP goal per output, in the order
    // given), each constrained by the committed plan of the earlier
    // rounds (a committed tensor may only produce its committed split),
    // and commit the round's new parameter splits to the meta device's
    // table (the earlier rounds' entries are untouched). The returned
    // plan is the cumulative plan (last_plan_ + the round's new nodes);
    // an infeasible round rolls back its tentative commits and returns
    // the previous cumulative plan marked infeasible, so the allocation
    // built against the committed plan stays valid.
    Plan plan_round(const std::vector<Tensor>& outputs);

    MetaDevice& device_;                 // the shared split-state table + device count
    double w_comm_;

    // The single trace of everything: one ShardingRuntime traced every
    // context's forward (topological order), owned by the allocator.
    ShardingRuntime runtime_;

    // The committed plan of every finished round: trace node -> the split
    // it was planned with, locked for every later output and round. The
    // DP treats a committed tensor as fixed (exact(): only its committed
    // distribution is feasible; best(): only it is producible), which is
    // what keeps the incremental plans consistent with the split states
    // already materialized in the meta device's table.
    struct Committed {
        ShardingRuntime::Dist produced;   // the distribution the node produces
        ExactState exact;                 // the state it was planned with (feasible, cost, cand, in_dists)
    };
    std::map<int, Committed> committed_;
    
    // The node ids committed since the start of the current round: an
    // infeasible round rolls them back (a failed round must not lock in
    // any of its decisions).
    std::vector<int> committed_this_round_;
    
    // Lock one newly planned node into committed_ (the DP of the next
    // output of this round -- and of every later round -- must plan
    // around it). Called in execution order (inputs before consumers),
    // so a node's inputs are already committed when its state is locked
    // in.
    void commit_node(const PlanNode& pn);
    
    // Undo the current round's tentative commits (an infeasible round).
    void rollback_round();
    
    // The committed parameter splits of every finished round (the public
    // view of committed_): one entry per shared parameter, accumulated
    // over the rounds.
    std::map<const ggml_tensor*, ShardingRuntime::Dist> decisions_;

    // The trace ids of this round's goal roots (the graph outputs): they
    // must be produced exactly in the required state, never bridged
    // (see best()).
    std::set<int> goal_roots_;

    std::map<int, std::map<ShardingRuntime::Dist, ExactState>> exact_memo_;
    std::map<int, std::map<ShardingRuntime::Dist, BestState>> best_memo_;

    MetaDevice::Splits splits_;

    Plan last_plan_;
};
