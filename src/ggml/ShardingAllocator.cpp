#include "ggml/ShardingAllocator.hpp"
#include <functional>

void ShardingAllocator::allocate(const std::vector<Context*>& contexts, const std::vector<Tensor>& outputs) {
    // One allocation round: plan the outputs incrementally (the outputs an
    // earlier round did not plan; each DP is constrained by the committed
    // plan of the earlier rounds) -- commit the round's new parameter
    // splits to the meta device's table -- then run the base allocation
    // over the round's contexts.

    last_plan_ = plan_round(outputs);

    std::cerr << last_plan_.to_string();

    if (last_plan_.infeasible) {
        // The trace shows every node and the output distributions its
        // candidates can produce -- the way in to see why the DP gave up.
        throw std::runtime_error(
            "ShardingAllocator: the allocation plan is infeasible: " + last_plan_.infeasible_reason + "\n\n" + dump_trace());
    }

    splits_ = device_.splits();

    // The base allocation is the incremental counterpart of the
    // incremental plan: called again with the round's contexts, it
    // allocates only the tensors without a buffer yet.
    Allocator::allocate(contexts, outputs);
}

void ShardingAllocator::commit_node(const PlanNode& pn) {
    if (committed_.count(pn.id))
        return;   // already committed (an earlier round, or twice in this one)

    Committed c;
    c.produced = pn.produced;
    // The inputs are committed already (execution order), so the state is
    // the one the DP planned with -- locked in as-is.
    c.exact = exact(pn.id, pn.produced);
    committed_.emplace(pn.id, std::move(c));
    committed_this_round_.push_back(pn.id);
}

void ShardingAllocator::rollback_round() {
    for (int id : committed_this_round_)
        committed_.erase(id);
    committed_this_round_.clear();
    goal_roots_.clear();
}

ShardingAllocator::Plan ShardingAllocator::plan_round(const std::vector<Tensor>& outputs) {
    // The plan is incremental: last_plan_ (and committed_) hold every
    // tensor the earlier allocation rounds planned, locked to its
    // committed split. This round adds the outputs that are not planned
    // yet -- one DP goal per new output, in the order given, each
    // constrained by the committed plan (a committed tensor may only
    // produce its committed distribution; an output an earlier round
    // planned is skipped -- the DP is solved for it exactly once, when it
    // first becomes a root). The DP is strictly acyclic (topological
    // order), so the memoized recursion terminates.

    Plan plan = last_plan_;
    plan.infeasible = false;
    plan.infeasible_reason.clear();

    if (runtime_.nodes().empty() || outputs.empty())
        return plan;

    plan.device_count = device_.count();
    const std::vector<ShardingRuntime::TraceNode>& nodes = runtime_.nodes();
    const std::vector<ggml_tensor*>& raw = runtime_.raw_of();

    std::set<std::pair<int, ShardingRuntime::Dist>> emitted;
    for (const auto& root : outputs) {
        const int root_id = runtime_.id_of(*root);

        // Planned by an earlier round (or an earlier output of this
        // round): the DP is already solved for it -- skip. The base
        // allocation below still runs: it places this round's new tensors
        // against the committed table.
        if (committed_.count(root_id))
            continue;

        const Goal g = {root_id, ShardingRuntime::Dist::replicated()};
        goal_roots_.clear();
        goal_roots_.insert(root_id);
        exact_memo_.clear();
        best_memo_.clear();
        const auto& root_state = best(root_id, g.required);   // roots are exact-only (no bridge)
        if (!root_state.feasible) {
            plan.infeasible = true;
            plan.infeasible_reason = infeasibility_reason(g.required);
            rollback_round();
            return plan;
        }

        const size_t before = plan.nodes.size();
        emit(root_id, g.required, plan, emitted);

        // The meta backend derives exactly one split state per tensor, so
        // a tensor this round emitted in two different distributions
        // cannot be planned (one storage layout per tensor) -- within
        // this round as well as against the committed plan.
        {
            std::map<int, ShardingRuntime::Dist> round_states;
            for (size_t i = before; i < plan.nodes.size(); ++i) {
                const PlanNode& pn = plan.nodes[i];
                auto [it, inserted] = round_states.insert({pn.id, pn.produced});
                if (!inserted && it->second != pn.produced) {
                    plan.infeasible = true;
                    plan.infeasible_reason = nodes[pn.id].op_name + " (node " + std::to_string(pn.id) +
                        ") is required in both " + it->second.to_string() + " and " +
                        pn.produced.to_string() + ", but the meta backend derives a single state per tensor";
                    rollback_round();
                    return plan;
                }
            }
        }

        // Lock the new nodes: the next output of this round -- and every
        // later round -- must plan around them. Reverse (execution)
        // order: a node's inputs before the node itself, so the inputs
        // are committed when the node's state is locked in.
        for (size_t i = plan.nodes.size(); i-- > before; )
            commit_node(plan.nodes[i]);
    }

    // Commit the round's new parameter splits to the meta device's split
    // table -- the shared resource, global across contexts and allocators.
    // The earlier rounds' entries are untouched: they are locked in by
    // the allocation built against them. The round's true cost: every new
    // tensor is planned (and paid for) exactly once, plus its P -> R
    // bridge (the DP above pays a shared input once per consumer, so its
    // total overcounts such subtrees).
    double cost = 0.0;
    for (size_t i = last_plan_.nodes.size(); i < plan.nodes.size(); ++i) {
        const PlanNode& pn = plan.nodes[i];
        if (nodes[pn.id].is_param) {
            plan.callback_dists[pn.id] = pn.produced;
            decisions_[raw[pn.id]] = pn.produced;
            const ggml_backend_meta_split_state st = materialize(pn.produced, nodes[pn.id].ne, raw[pn.id]->type);
            plan.callback_states[param_name(pn.id)] = st;
            device_.splits()[raw[pn.id]] = st;
        }

        const ExactState& e = exact(pn.id, pn.produced);
        if (e.cand >= 0)
            cost += nodes[pn.id].candidates[e.cand].comp_cost;
        cost += pn.bridge_cost;
    }
    plan.total_cost += cost;

    committed_this_round_.clear();
    goal_roots_.clear();
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
