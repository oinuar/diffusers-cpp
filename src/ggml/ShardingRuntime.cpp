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
