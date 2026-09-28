#include "ggml/ShardingRuntime.hpp"

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
// ============================================================================
// Planning: one DP over the WHOLE trace
// ============================================================================
bool ShardingRuntime::plan(Plan** plan) {
    // Solves the WHOLE trace in one go: a single DP over every traced
    // tensor, the goal roots being the outputs the trace marked with
    // set_output() (goal_roots()), each required exactly replicated (R)
    // -- the only state a graph output can be read back in. The roots
    // share the one DP memo, so a tensor shared by several roots (a
    // parameter consumed by several contexts' forwards) is ONE node,
    // planned exactly once, for all of them. The DP is strictly acyclic
    // (topological order), so the memoized recursion terminates.

    Plan new_plan;
    new_plan.device_count = device_.count();

    const std::set<int>& roots = goal_roots_;
    if (nodes_.empty() || roots.empty()) {
        new_plan.infeasible = true;
        new_plan.infeasible_reason = "the trace has no outputs marked with set_output() (nothing to plan for)";
        last_plan_ = new_plan;
        *plan = &last_plan_.value();
        return true;
    }

    // A re-plan with an unchanged trace and cost model reuses the last
    // plan (the DP is not re-run): repeated allocations with the same
    // trace do not re-solve it.
    if (last_plan_ && !last_plan_->infeasible && planned_trace_size_ == nodes_.size() && planned_roots_ == roots) {
        *plan = &last_plan_.value();
        return false;
    }

    const Dist root_dist = Dist::replicated();

    // One DP for the whole trace: the single memo is shared by every goal
    // root (a shared tensor is planned once, for all of them).
    exact_memo_.clear();
    best_memo_.clear();
    for (const int root : roots) {
        const BestState& root_state = best(root, root_dist);   // roots are exact-only (no bridge)
        if (!root_state.feasible) {
            new_plan.infeasible = true;
            new_plan.infeasible_reason = infeasibility_reason(root_dist);
            last_plan_ = new_plan;
            *plan = &last_plan_.value();
            return true;
        }
    }

    // The plan itself: DFS from every goal root. A tensor is planned
    // (emitted) exactly once, in the distribution its first consumer
    // needs, and every later consumer is served from that same state.
    std::set<std::pair<int, Dist>> emitted;
    for (const int root : roots)
        emit(root, root_dist, new_plan, emitted);

    // The meta backend derives exactly one split state per tensor (its
    // storage layout), so a tensor the plan emitted with two different
    // PRODUCED distributions cannot be planned. (A produced distribution
    // bridged to a different required one is fine: the bridge is a
    // collective at the consumer, not a second storage state.)
    {
        std::map<int, Dist> produced;
        for (const PlanNode& pn : new_plan.nodes) {
            auto [it, inserted] = produced.insert({pn.id, pn.produced});
            if (!inserted && it->second != pn.produced) {
                new_plan.infeasible = true;
                new_plan.infeasible_reason = nodes_[pn.id].op_name + " (node " + std::to_string(pn.id) +
                    ") would need two different storage states (" + it->second.to_string() + " and " +
                    pn.produced.to_string() + "), but the meta backend derives a single state per tensor";
                last_plan_ = new_plan;
                *plan = &last_plan_.value();
                return true;
            }
        }
    }

    // The parameter splits: one entry per shared parameter (the meta
    // backend derives the compute tensors' states). Retires the splits of
    // the previous plan (its raw tensors may have been re-created by a
    // re-trace) and commits the new ones to the meta device's split table
    // -- the shared resource, global across contexts and allocators. The
    // true cost: every planned tensor is paid exactly once, plus its P ->
    // R bridge (the DP above pays a shared input once per consumer, so
    // its total overcounts such subtrees).
    for (const auto& [t, st] : splits_)
        device_.splits().erase(t);
    splits_.clear();

    decisions_.clear();
    new_plan.callback_states.clear();

    double cost = 0.0;
    for (const PlanNode& pn : new_plan.nodes) {
        if (nodes_[pn.id].is_param) {
            const ggml_tensor* raw = raw_of_[pn.id];

            decisions_[raw] = pn.produced;
            const ggml_backend_meta_split_state st = materialize(pn.produced, nodes_[pn.id].ne, raw->type);
            new_plan.callback_states[param_name(pn.id)] = st;
            device_.splits()[raw] = st;
            splits_[raw] = st;
        }

        const ExactState& e = exact(pn.id, pn.produced);
        if (e.cand >= 0)
            cost += nodes_[pn.id].candidates[e.cand].comp_cost;
        cost += pn.bridge_cost;
    }
    new_plan.total_cost = cost;

    planned_trace_size_ = nodes_.size();
    planned_roots_ = roots;

    last_plan_ = new_plan;
    *plan = &last_plan_.value();
    return true;
}

std::string ShardingRuntime::dump_trace() const {
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
            ss << " " << c.output.to_string();
        if (n.candidates.empty())
            ss << " (none -- unsupported by the meta backend)";
        ss << "\n";
    }
    return ss.str();
}
