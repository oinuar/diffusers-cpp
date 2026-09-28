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
    }

    Allocator::allocate(context);
}
