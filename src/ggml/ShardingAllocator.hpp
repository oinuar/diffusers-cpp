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
// plan() solves the WHOLE trace in one go: a single DP over every traced
// tensor decides ONE sharding state per tensor, so the resulting plan
// works in the whole computation (every context's forward at once). The
// goal roots are the outputs the trace marked with set_output() while the
// graphs were built (the caller never names them to the planner): each
// root must be produced exactly replicated (R) -- the only state a graph
// output can be read back in. A tensor shared by several roots (a
// parameter consumed by several contexts' forwards) is ONE trace node and
// is planned exactly once, for all of them: the roots share the single DP
// memo, and the plan rejects a tensor needed in two different
// distributions (the meta backend derives one state per tensor).
//
// allocate(contexts, outputs) is the one-shot allocation: it plans the
// whole trace (plan()) -- the `outputs` argument is not needed by the
// planner -- commits the plan's parameter splits to the shared resource --
// the MetaDevice's split table, GLOBAL across contexts and across every
// allocator (the splits are just GGML's way of doing the sharding; the
// table is what the meta backend queries at runtime) -- then runs the base
// allocation over the contexts (it allocates only the tensors without a
// buffer yet). An infeasible plan commits nothing and allocate() throws:
// the previously committed table stays valid.
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
    ShardingAllocator(Runtime& parent, MetaDevice& device, double w_comp, double w_mem, double w_comm, const std::optional<ggml_backend_buffer_usage>& usage = std::nullopt)
        : Allocator(device, usage), device_(device), w_comm_(w_comm), runtime_(parent, device, w_comp, w_mem) {}

    virtual ~ShardingAllocator() = default;

    // The communication cost of the next plan (a re-plan with a changed
    // cost model).
    void set_w_comm(double w_comm) { w_comm_ = w_comm; }

    // The one-shot allocation: plan the whole trace (plan()) -- the goal
    // roots are the outputs the trace marked with set_output(), never the
    // `outputs` argument -- then run the base allocation over the
    // contexts (it allocates only the tensors without a buffer yet,
    // exactly like the base Allocator). An infeasible plan throws.
    void allocate(Context& context) override;

    // Solves the WHOLE trace in one go: one DP over every traced tensor,
    // the goal roots being the outputs the trace marked with set_output()
    // (each required exactly in R; a tensor shared by several roots is
    // planned exactly once, for all of them). Returns the plan: the
    // distribution every planned tensor is produced in, the P -> R
    // bridges, and the meta device's callback states for the params (the
    // plan's parameter splits are committed to the device's split table).
    // A re-plan with an unchanged trace and cost model reuses the last
    // plan (the DP is not re-run); an infeasible trace returns the plan
    // marked infeasible (nothing is committed).
    Plan plan();

    // The plan of the last plan() / allocate().
    const Plan& last_plan() const { return last_plan_; }

    // Debug dump of the trace: every node with its shape and the
    // output distributions its candidates can produce.
    std::string dump_trace() const;

    // The parameter splits of the last plan(): one entry per shared
    // parameter.
    const std::map<const ggml_tensor*, ShardingRuntime::Dist>& decisions() const { return decisions_; }

    const MetaDevice& device() const { return device_; }

private:
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
    // Tree DP over the whole trace, solved once per plan() (one go): a
    // single memo shared by every goal root, so F(node, d) pays every
    // shared input once per consumer (a sound bound, used only to select
    // a plan); the plan recomputes the emitted plan's true per-tensor
    // cost. A tensor shared by several roots (a param consumed by several
    // contexts' forwards) is ONE node here, so its storage is paid exactly
    // once, for all of them, and its split is decided exactly once.
    //
    // F(node, d): node produces exactly d.
    ExactState& exact(int node, const ShardingRuntime::Dist& d) {
        auto& m = exact_memo_[node][d];
        if (m.done) return m;
        m.done = true;

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
    // A goal root (an output marked with set_output(), see
    // ShardingRuntime::goal_roots()) must be produced
    // exactly in d: the only bridge (P -> R, the AllReduce) is
    // materialized by the meta backend only at a subgraph boundary before
    // a consumer, and its ggml_backend_meta_buffer_get_tensor (assume_sync
    // = false) has no PARTIAL case, so a PARTIAL output can never be read.
    BestState& best(int node, const ShardingRuntime::Dist& d) {
        auto& m = best_memo_[node][d];
        if (m.done) return m;
        m.done = true;

        const bool exact_only = runtime_.goal_roots().count(node) != 0;

        // A root is producible only in d (exact-only); every other node in
        // every candidate output.
        std::set<ShardingRuntime::Dist> producible;
        if (exact_only)
            producible.insert(d);
        else
            for (const ShardingRuntime::Candidate& cand : runtime_.nodes()[node].candidates)
                producible.insert(cand.output);

        for (const ShardingRuntime::Dist& p : producible) {
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
        // A tensor consumed several times (even by different roots) in the
        // same distribution is planned once. A tensor needed in two
        // different distributions would get two split states, and plan()
        // rejects that: the meta backend derives exactly one state per
        // tensor.
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
            // That S(1) split must be uniform across the devices, so the
            // heads dim (axis 1 of the flash_attn output) must divide by the
            // device count.
            else if (n.op_name == "flash_attn" && n.ne[1] % device_.count() != 0)
                r.push_back(n.op_name + " (node " + std::to_string(n.id) + ") " +
                    "must be sharded along the heads axis (S(1)), but its size " +
                    std::to_string(n.ne[1]) + " is not divisible by the device count " +
                    std::to_string(device_.count()));
        }

        // Infeasibility is because of unknown reason.
        if (r.empty())
            r.push_back("no feasible split plan satisfies the required output distribution " + required.to_string());

        return std::accumulate(std::begin(r), std::end(r), std::string(),
            [](const std::string& acc, const std::string& x) {
                if (acc.empty())
                    return x;

                return acc + "\n" + x;
            }
        );
    }

    MetaDevice& device_;                 // the shared split-state table + device count
    double w_comm_;

    // The single trace of everything: one ShardingRuntime traced every
    // context's forward (topological order), owned by the allocator.
    ShardingRuntime runtime_;

    // The parameter splits of the last plan(): one entry per shared
    // parameter (the public view via decisions()).
    std::map<const ggml_tensor*, ShardingRuntime::Dist> decisions_;

    std::map<int, std::map<ShardingRuntime::Dist, ExactState>> exact_memo_;
    std::map<int, std::map<ShardingRuntime::Dist, BestState>> best_memo_;

    MetaDevice::Splits splits_;

    Plan last_plan_;

    // The fingerprint of the last successful plan(): the trace's node
    // count, its goal roots, and the cost model. A re-plan with an
    // unchanged fingerprint reuses last_plan_ instead of re-running the
    // DP (repeated allocations with the same trace).
    bool planned_ = false;
    size_t planned_trace_size_ = 0;
    std::set<int> planned_roots_;
    double planned_w_comm_ = 0.0;
};
