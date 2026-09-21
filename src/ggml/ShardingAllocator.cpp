#include "ggml/ShardingAllocator.hpp"
#include <functional>

void ShardingAllocator::allocate(Context& context) {
    // The one-shot allocation: plan the WHOLE trace in one go (plan()) --
    // the goal roots are the outputs the trace marked with set_output()
    // while the graphs were built, never the `outputs` argument -- commit
    // the plan's parameter splits to the meta device's split table (done
    // in plan()) -- then run the base allocation over the contexts (it
    // allocates only the tensors without a buffer yet, exactly like the
    // base Allocator). An infeasible plan commits nothing and the
    // previously committed table stays valid.

    last_plan_ = plan();

    std::cerr << last_plan_.to_string();

    if (last_plan_.infeasible) {
        // The trace shows every node and the output distributions its
        // candidates can produce -- the way in to see why the DP gave up.
        throw std::runtime_error(
            "allocate(): the allocation plan is infeasible: " + last_plan_.infeasible_reason + "\n\n" + dump_trace());
    }

    Allocator::allocate(context);
}

ShardingAllocator::Plan ShardingAllocator::plan() {
    // Solves the WHOLE trace in one go: a single DP over every traced
    // tensor, the goal roots being the outputs the trace marked with
    // set_output() (ShardingRuntime::goal_roots()), each required exactly
    // replicated (R) -- the only state a graph output can be read back in.
    // The roots share the one DP memo, so a tensor shared by several
    // roots (a parameter consumed by several contexts' forwards) is ONE
    // node, planned exactly once, for all of them. The DP is strictly
    // acyclic (topological order), so the memoized recursion terminates.

    Plan plan;
    plan.device_count = device_.count();

    const std::set<int>& roots = runtime_.goal_roots();
    if (runtime_.nodes().empty() || roots.empty()) {
        plan.infeasible = true;
        plan.infeasible_reason = "the trace has no outputs marked with set_output() (nothing to plan for)";
        last_plan_ = plan;
        return plan;
    }

    // A re-plan with an unchanged trace and cost model reuses the last
    // plan (the DP is not re-run): repeated allocations with the same
    // trace do not re-solve it.
    if (planned_ && planned_trace_size_ == runtime_.nodes().size() && planned_roots_ == roots && planned_w_comm_ == w_comm_)
        return last_plan_;

    const ShardingRuntime::Dist root_dist = ShardingRuntime::Dist::replicated();

    // One DP for the whole trace: the single memo is shared by every goal
    // root (a shared tensor is planned once, for all of them).
    exact_memo_.clear();
    best_memo_.clear();
    for (const int root : roots) {
        const BestState& root_state = best(root, root_dist);   // roots are exact-only (no bridge)
        if (!root_state.feasible) {
            plan.infeasible = true;
            plan.infeasible_reason = infeasibility_reason(root_dist);
            last_plan_ = plan;
            return plan;
        }
    }

    // The plan itself: DFS from every goal root. A tensor is planned
    // (emitted) exactly once, in the distribution its first consumer
    // needs, and every later consumer is served from that same state.
    std::set<std::pair<int, ShardingRuntime::Dist>> emitted;
    for (const int root : roots)
        emit(root, root_dist, plan, emitted);

    // The meta backend derives exactly one split state per tensor (its
    // storage layout), so a tensor the plan emitted with two different
    // PRODUCED distributions cannot be planned. (A produced distribution
    // bridged to a different required one is fine: the bridge is a
    // collective at the consumer, not a second storage state.)
    {
        std::map<int, ShardingRuntime::Dist> produced;
        for (const PlanNode& pn : plan.nodes) {
            auto [it, inserted] = produced.insert({pn.id, pn.produced});
            if (!inserted && it->second != pn.produced) {
                plan.infeasible = true;
                plan.infeasible_reason = runtime_.nodes()[pn.id].op_name + " (node " + std::to_string(pn.id) +
                    ") would need two different storage states (" + it->second.to_string() + " and " +
                    pn.produced.to_string() + "), but the meta backend derives a single state per tensor";
                last_plan_ = plan;
                return plan;
            }
        }
    }

    // The parameter splits: one entry per shared parameter (the meta
    // backend derives the compute tensors' states). Retires the splits of
    // this allocator's previous plan (its raw tensors may have been
    // re-created by a re-trace) and commits the new ones to the meta
    // device's split table -- the shared resource, global across contexts
    // and allocators. The true cost: every planned tensor is paid exactly
    // once, plus its P -> R bridge (the DP above pays a shared input once
    // per consumer, so its total overcounts such subtrees).
    for (const auto& [t, d] : splits_)
        device_.splits().erase(t);
    splits_.clear();

    decisions_.clear();
    plan.callback_dists.clear();
    plan.callback_states.clear();

    double cost = 0.0;
    for (const PlanNode& pn : plan.nodes) {
        if (runtime_.nodes()[pn.id].is_param) {
            const ggml_tensor* raw = runtime_.raw_of()[pn.id];

            plan.callback_dists[pn.id] = pn.produced;
            decisions_[raw] = pn.produced;
            const ggml_backend_meta_split_state st = materialize(pn.produced, runtime_.nodes()[pn.id].ne, raw->type);
            plan.callback_states[param_name(pn.id)] = st;
            device_.splits()[raw] = st;
            splits_[raw] = st;
        }

        const ExactState& e = exact(pn.id, pn.produced);
        if (e.cand >= 0)
            cost += runtime_.nodes()[pn.id].candidates[e.cand].comp_cost;
        cost += pn.bridge_cost;
    }
    plan.total_cost = cost;

    planned_ = true;
    planned_trace_size_ = runtime_.nodes().size();
    planned_roots_ = roots;
    planned_w_comm_ = w_comm_;

    last_plan_ = plan;
    return plan;
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
