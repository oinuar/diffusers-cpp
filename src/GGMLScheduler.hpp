#pragma once

#include "GGMLCompute.hpp"
#include "GGMLArena.hpp"
#include "GGMLGraph.hpp"
#include "GGMLContext.hpp"

#include "ggml.h"
#include "ggml-backend.h"

#include <vector>

class GGMLScheduler {
public:
    GGMLScheduler(std::vector<ggml_backend_t>&& backends, size_t graph_size = GGML_DEFAULT_GRAPH_SIZE)
        : backends_(std::move(backends)), arena_(graph_size)
    {
        sched_ = ggml_backend_sched_new(backends_.data(), nullptr, backends_.size(), graph_size, false, true);
    }

    GGMLScheduler(GGMLScheduler&& other)
        : backends_(std::move(other.backends_)), arena_(std::move(other.arena_)), sched_(other.sched_)
    {
        other.sched_ = nullptr;
    }

    ~GGMLScheduler() {
        if (sched_ != nullptr)
            ggml_backend_sched_free(sched_);
    }

    GGMLGraph plan(GGMLCompute& compute) {
        GGMLContext ctx(arena_);

        auto gf = ggml_new_graph(*ctx);
        auto result = compute.build(ctx);

        return std::move(GGMLGraph(gf, sched_, result));
    }

    GGMLScheduler(GGMLScheduler&) = delete;
    GGMLScheduler& operator =(const GGMLScheduler&) = delete;
private:
    std::vector<ggml_backend_t> backends_;
    GGMLArena arena_;
    ggml_backend_sched_t sched_;
};
