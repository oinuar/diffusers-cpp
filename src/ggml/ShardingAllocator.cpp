#include "ggml/ShardingAllocator.hpp"
#include <iostream>

void ShardingAllocator::allocate(Context& context) {
    ShardingRuntime::Plan* plan;

    if (runtime_.plan(&plan)) {
        if (plan->infeasible) {
            // The trace shows every node and the output distributions its
            // candidates can produce -- the way in to see why the DP gave up.
            throw std::runtime_error(
                "allocate(): the allocation plan is infeasible: " + plan->infeasible_reason + "\n\n" + runtime_.dump_trace());
        }

        std::cerr << plan->to_string();

        auto total_sharded_bytes = 0.0;
        auto total_replicated_bytes = 0.0;

        for (auto& node : plan->nodes) {
            auto bytes = ggml_nbytes(runtime_.raw_of()[node.id]);

            if (node.produced.type == ShardingRuntime::Dist::R)
                total_replicated_bytes += bytes;
            else
                total_sharded_bytes += bytes;
        }

        std::cerr << "Required memory for the plan: "
                  << format_bytes(total_sharded_bytes)
                  << " (sharded) + "
                  << format_bytes(total_replicated_bytes)
                  << " (replicated) = "
                  << format_bytes(total_sharded_bytes + total_replicated_bytes)
                  << std::endl;
    }
    
    Allocator::allocate(context);
}
