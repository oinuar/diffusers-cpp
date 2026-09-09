#pragma once

#include "ggml/Allocator.hpp"
#include "ggml/ShardingEngine.hpp"
#include <set>
#include <sstream>
#include <iomanip>

// ============================================================================
// ShardedAllocator -- the sharded version of the Allocator: the planning
// state
// (drafted in src/ggml/ShardedAllocator.hpp)
//
// The allocation is WHY we plan: we plan to allocate the tensors optimally
// across the devices. The allocator OWNS the trace: it constructs the ONE
// ShardingEngine every context's forward() runs through, so every
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

class ShardedAllocator : public Allocator {
public:

    struct PlanNode {
        int id = 0;
        std::string op_name;
        std::string tensor_name;        // non-empty for params (the callback key)
        ShardingEngine::Dist produced;
        ShardingEngine::Dist required;
        std::string bridge;                 // collective between produced and required
        double bridge_cost = 0.0;
    };

    struct Plan {
        double total_cost = 0.0;
        size_t device_count = 0;   // for printing the per-device split sizes
        bool infeasible = false;
        std::string infeasible_reason;
        std::vector<PlanNode> nodes;        // DFS preorder; printed in reverse = execution order
        std::map<int, ShardingEngine::Dist> callback_dists; // param node id -> storage distribution
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
    // constructs the ONE ShardingEngine over it -- the trace every forward
    // runs through (Scope over allocator.engine()). `device` is the shared
    // resource this allocator commits its plans to: the global split table
    // (plus the device count). `w_comp`/`w_mem` shape the candidates the
    // engine generates; `w_comm` prices the P -> R bridge (see the cost
    // model in the file header).
    ShardedAllocator(Engine& parent, MetaDevice& device, double w_comp, double w_mem, double w_comm)
        : device_(device), w_comm_(w_comm), engine_(parent, device, w_comp, w_mem) {}

    virtual ~ShardedAllocator() = default;

    // The engine every forward runs through: the trace the allocator plans.
    ShardingEngine& engine() override { return engine_; }

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
    void allocate(const std::vector<Tensor>& outputs, bool reallocate = false) override;

    bool is_stale() const;

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
    const std::map<const ggml_tensor*, ShardingEngine::Dist>& decisions() const { return decisions_; }

    const MetaDevice& device() const { return device_; }

private:
    struct Goal {
        int root;
        ShardingEngine::Dist required;
    };

    struct Bridge {
        std::string name;
        double cost;
    };

    // F(node, d): node produces exactly d.
    struct ExactState {
        bool done = false;
        double cost = ShardingEngine::kInf;
        int cand = -1;
        std::vector<ShardingEngine::Dist> in_dists;
    };

    // G(node, d): node satisfies d (produces some d' and bridges d' -> d).
    struct BestState {
        bool done = false;
        double cost = ShardingEngine::kInf;
        ShardingEngine::Dist produced;
    };

    // The collective needed to turn a tensor in `from` into the distribution
    // `to`, and its per-device cost. The meta backend's only collective is
    // the AllReduce at a PARTIAL subgraph boundary -- there is no
    // AllGather/ReduceScatter/AllToAll, so everything except P -> R is
    // infeasible. A sharded tensor is consumed sharded through the per-op
    // rules; a full tensor is (re-)produced by a row-parallel mul_mat +
    // the implicit AllReduce.
    Bridge bridge(const ShardingEngine::Dist& from, const ShardingEngine::Dist& to) const {
        if (from == to) return {"None", 0.0};
        if (from.type == ShardingEngine::Dist::Type::P && to.type == ShardingEngine::Dist::Type::R)
            return {"AllReduce", 0.5 * w_comm_ * comm_factor()};
        return {"Infeasible", ShardingEngine::kInf};
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
    ggml_backend_meta_split_state materialize(const ShardingEngine::Dist& d, const int64_t ne[4], ggml_type type) const {
        ggml_backend_meta_split_state st;
        std::memset(&st, 0, sizeof(st));
        st.axis = d.to_split_axis();
        st.nr[0] = 1;
        st.n_segments = 1;
        if (d.type == ShardingEngine::Dist::Type::S) {
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
        const char* n = engine_.raw_of()[id]->name;
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
    ExactState& exact(int node, const ShardingEngine::Dist& d) {
        auto& m = exact_memo_[node][d];
        if (m.done) return m;
        m.done = true;

        const ShardingEngine::TraceNode& n = engine_.nodes()[node];
        // A parameter whose split is already committed (planned by an
        // earlier output in this plan round) must keep it: the meta
        // backend derives one state per tensor, shared by every output
        // that consumes it.
        if (n.is_param) {
            const auto c = decisions_.find(engine_.raw_of()[node]);
            if (c != decisions_.end() && c->second != d)
                return m;   // infeasible state (cost stays kInf)
        }
        for (int c = 0; c < (int)n.candidates.size(); ++c) {
            const ShardingEngine::Candidate& cand = n.candidates[c];
            if (cand.output != d) continue;

            double cost = cand.comp_cost;
            bool ok = true;
            std::vector<ShardingEngine::Dist> ins;
            ins.reserve(cand.inputs.size());
            for (size_t i = 0; i < cand.inputs.size(); ++i) {
                const double in_cost = best(n.inputs[i], cand.inputs[i]).cost;
                if (in_cost >= ShardingEngine::kInf / 2) { ok = false; break; }
                cost += in_cost;
                ins.push_back(cand.inputs[i]);
            }
            if (ok && cost < m.cost)
                m = {true, cost, c, std::move(ins)};
        }
        return m;
    }

    // G(node, d): node satisfies d -- produce some producible d', then bridge.
    BestState& best(int node, const ShardingEngine::Dist& d) {
        auto& m = best_memo_[node][d];
        if (m.done) return m;
        m.done = true;

        std::set<ShardingEngine::Dist> producible;
        for (const ShardingEngine::Candidate& cand : engine_.nodes()[node].candidates)
            producible.insert(cand.output);

        for (const ShardingEngine::Dist& p : producible) {
            const double exact_cost = exact(node, p).cost;
            if (exact_cost >= ShardingEngine::kInf / 2) continue;
            const Bridge b = bridge(p, d);
            if (b.cost >= ShardingEngine::kInf / 2) continue;
            const double total = exact_cost + b.cost;
            if (total < m.cost)
                m = {true, total, p};
        }
        return m;
    }

    void emit(int node, const ShardingEngine::Dist& required, Plan& plan, std::set<std::pair<int, ShardingEngine::Dist>>& emitted) {
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
        pn.op_name = engine_.nodes()[node].op_name;
        pn.tensor_name = engine_.nodes()[node].is_param ? param_name(node) : "";
        pn.produced = b.produced;
        pn.required = required;
        pn.bridge = std::move(br.name);
        pn.bridge_cost = br.cost;
        plan.nodes.push_back(std::move(pn));

        const ExactState& e = exact(node, b.produced);
        if (e.cand < 0) return;
        const ShardingEngine::Candidate& cand = engine_.nodes()[node].candidates[e.cand];
        const ShardingEngine::TraceNode& n = engine_.nodes()[node];
        for (size_t i = 0; i < cand.inputs.size(); ++i)
            emit(n.inputs[i], cand.inputs[i], plan, emitted);
    }

    std::string infeasibility_reason(const ShardingEngine::Dist& required) const {
        std::string r;
        for (const ShardingEngine::TraceNode& n : engine_.nodes()) {
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
    Plan plan_round(const std::vector<Tensor>& outputs);

    MetaDevice& device_;                 // the shared split-state table + device count
    double w_comm_;

    // The single trace of everything: one ShardingEngine traced every
    // context's forward (topological order), owned by the allocator.
    ShardingEngine engine_;

    // The roots of the last committed plan: allocate() plans only when
    // the outputs differ from these -- a new graph, or a fresh trace
    // after forget(); the same outputs (the same graph computed again)
    // skip the DP.
    std::vector<Tensor> planned_outputs_;

    std::map<const ggml_tensor*, ShardingEngine::Dist> decisions_;   // committed param splits, accumulated as the round plans the outputs

    std::map<int, std::map<ShardingEngine::Dist, ExactState>> exact_memo_;
    std::map<int, std::map<ShardingEngine::Dist, BestState>> best_memo_;

    MetaDevice::Splits splits_;

    Plan last_plan_;
};
