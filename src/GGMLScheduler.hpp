#pragma once

#include "GGMLComputation.hpp"
#include "GGMLArena.hpp"
#include "GGMLGraph.hpp"
#include "GGMLContext.hpp"

#include "ggml.h"
#include "ggml-backend.h"

#include <vector>

class GGMLScheduler {
public:
    GGMLScheduler(std::vector<ggml_backend_t>&& backends)
        : backends_(std::move(backends)), arena_(GGML_DEFAULT_GRAPH_SIZE)
    {
        sched_ = ggml_backend_sched_new(backends_.data(), nullptr, backends_.size(), GGML_DEFAULT_GRAPH_SIZE, false, true);
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

    GGMLGraph plan(GGMLComputation& computation) {
        GGMLContext ctx(arena_);

        auto gf = ggml_new_graph(*ctx);
        auto [deps, result] = computation.build(ctx);

        return std::move(GGMLGraph(gf, sched_, std::move(deps), result));
    }

    GGMLScheduler(GGMLScheduler&) = delete;
    GGMLScheduler& operator =(const GGMLScheduler&) = delete;
private:
    std::vector<ggml_backend_t> backends_;
    GGMLArena arena_;
    ggml_backend_sched_t sched_;
};

class GGMLScope {
public:
    GGMLScope(GGMLGraph& graph) : graph_(graph) {
        ggml_backend_sched_reset(graph.sched_);
        ggml_backend_sched_alloc_graph(graph.sched_, graph.gf_);
    }

    ~GGMLScope() {
        ggml_backend_sched_graph_compute(graph_.sched_, graph_.gf_);
    }

    std::vector<Tensor>& inputs() {
        return graph_.inputs_;
    }

    GGMLScope(GGMLScope&) = delete;
    GGMLScope(GGMLScope&&) = delete;
    GGMLScope& operator =(const GGMLScope&) = delete;
    GGMLScope& operator =(GGMLScope&&) = delete;

private:
    GGMLGraph& graph_;
};
