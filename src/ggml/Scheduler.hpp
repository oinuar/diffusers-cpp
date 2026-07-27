#pragma once

#include "Compute.hpp"
#include "Arena.hpp"
#include "Graph.hpp"
#include "Context.hpp"
#include "Runtime.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include <vector>
#include <random>

class Scheduler {
public:
    Scheduler(std::vector<ggml_backend_t>&& backends, uint64_t seed = std::random_device{}(), size_t graph_size = GGML_DEFAULT_GRAPH_SIZE)
        : backends_(std::move(backends)), arena_(graph_size), seed_(seed)
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
        Runtime runtime(ctx, seed_);

        auto gf = ggml_new_graph(*ctx);
        auto compute_plan = compute.build(runtime);

        return std::move(Graph(
            gf,
            sched_,
            std::move(runtime.rng_),
            std::move(runtime.inputs_),
            std::move(compute_plan.tensors_)
        ));
    }

    Scheduler(Scheduler&) = delete;
    Scheduler& operator =(const Scheduler&) = delete;
private:
    std::vector<ggml_backend_t> backends_;
    Arena arena_;
    ggml_backend_sched_t sched_;
    uint64_t seed_;
};
