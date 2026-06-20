#pragma once

#include "GGMLGraph.hpp"

#include "ggml.h"
#include "ggml-backend.h"

#include <vector>

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
