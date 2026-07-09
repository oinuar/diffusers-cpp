#pragma once

#include "Compute.hpp"
#include "Arena.hpp"
#include "Graph.hpp"
#include "Context.hpp"

#include "ggml.h"
#include "ggml-backend.h"

#include <vector>

class Scheduler {
public:
    Scheduler(std::vector<ggml_backend_t>&& backends, size_t graph_size = GGML_DEFAULT_GRAPH_SIZE)
        : backends_(std::move(backends)), arena_(graph_size)
    {
        sched_ = ggml_backend_sched_new(backends_.data(), nullptr, backends_.size(), graph_size, false, true);
    }

    Scheduler(Scheduler&& other)
        : backends_(std::move(other.backends_)), arena_(std::move(other.arena_)), sched_(other.sched_)
    {
        other.sched_ = nullptr;
    }

    ~Scheduler() {
        if (sched_ != nullptr)
            ggml_backend_sched_free(sched_);
    }

    Graph plan(Compute& compute) {
        Context ctx(arena_);

        auto gf = ggml_new_graph(*ctx);
        auto result = compute.build(ctx);

        // Materialize tensor if needed
        if (!result.is_contiguous())
            result = result.contiguous();

        ggml_set_output(*result);

        return std::move(Graph(gf, sched_, result));
    }

    Scheduler(Scheduler&) = delete;
    Scheduler& operator =(const Scheduler&) = delete;
private:
    std::vector<ggml_backend_t> backends_;
    Arena arena_;
    ggml_backend_sched_t sched_;
};
