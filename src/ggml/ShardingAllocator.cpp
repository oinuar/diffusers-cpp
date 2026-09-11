#include "ggml/ShardingAllocator.hpp"

void ShardingAllocator::forget(const Context& context) {
    unuse(context);         // free the context's buffers
    runtime_.reset();        // the trace spanned the context; its tensors go away
    planned_outputs_.clear();   // the plan is invalid (a re-plan re-traces the live contexts)
    decisions_.clear();
}

void ShardingAllocator::allocate(const std::vector<Tensor>& outputs, bool reallocate) {
    // Plan the output-tensor graph exactly once: the first round for this
    // output set runs the DP and commits the splits; a changed output set
    // (a different graph) replans; the same outputs (the same graph
    // computed again) skip the DP -- the committed plan already covers
    // them.

    auto replan = !outputs.empty() && !runtime_.nodes().empty() && (outputs != planned_outputs_ || is_stale());

    if (replan) {
        last_plan_ = plan_round(outputs);

        std::cerr << last_plan_.to_string();

        if (last_plan_.infeasible) {
            // The trace shows every node and the output distributions its
            // candidates can produce -- the way in to see why the DP gave up.
            throw std::runtime_error(dump_trace() + "\n" + last_plan_.infeasible_reason);
        }

        planned_outputs_ = outputs;
        splits_ = device_.splits();
    }

    // The base allocation places every registered context against the
    // device's current split table: the contexts whose snapshots went
    // stale for the split states this round committed are freed and
    // reallocated, the untouched contexts keep their buffers.
    Allocator::allocate(outputs, replan || reallocate);
}

// Strict comparison of two split states (the meta backend compares them
// field-wise, see split_states_equal in ggml-backend-meta.cpp). For the
// single-segment states the planner materializes, a strict comparison is
// exactly what the allocator's staleness check wants: any changed
// boundary means a different per-device size.
static bool split_state_equal(const ggml_backend_meta_split_state& a, const ggml_backend_meta_split_state& b) {
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

bool ShardingAllocator::is_stale() const {
    for (const auto& [t, state] : splits_)
        if (!split_state_equal(device_.split(t), state))
            return true;
    return false;
}

ShardingAllocator::Plan ShardingAllocator::plan_round(const std::vector<Tensor>& outputs) {
    decisions_.clear();

    Plan plan;
    if (runtime_.nodes().empty() || outputs.empty())
        return plan;

    plan.device_count = device_.count();
    const std::vector<ShardingRuntime::TraceNode>& nodes = runtime_.nodes();
    const std::vector<ggml_tensor*>& raw = runtime_.raw_of();

    // One goal per output, in graph order: the output's trace node must
    // end in R (the final result is usable on every device) -- and it
    // must end in R produced directly, with no bridge: the meta backend
    // materializes the P -> R AllReduce only at a subgraph boundary
    // before a consumer, and it cannot read a PARTIAL output at all
    // (best() enforces this through goal_roots_).
    // Each output's DP sees the splits committed by the earlier outputs'
    // plans as fixed (the meta backend derives one state per tensor,
    // shared by every output that consumes it), so the first output to
    // plan a parameter decides its split for all of them, and the later
    // outputs adapt to it. The DP is strictly acyclic (topological
    // order), so the memoized recursion terminates.
    goal_roots_.clear();
    for (const auto& root : outputs)
        goal_roots_.insert(runtime_.id_of(*root));

    std::set<std::pair<int, ShardingRuntime::Dist>> emitted;
    for (auto root : outputs) {
        const Goal g = {runtime_.id_of(*root), ShardingRuntime::Dist::replicated()};
        exact_memo_.clear();
        best_memo_.clear();
        const double total = best(g.root, g.required).cost;   // roots are exact-only (no bridge)
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
