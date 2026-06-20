#pragma once

#include "Tensor.hpp"
#include "ggml.h"
#include "ggml-backend.h"

#include <vector>

class GGMLGraph {
public:
    GGMLGraph(ggml_cgraph* gf, ggml_backend_sched_t sched, std::vector<Tensor>&& inputs, Tensor tensor)
        : gf_(gf), sched_(sched), inputs_(std::move(inputs)), tensor_(tensor)
    {
        ggml_build_forward_expand(gf_, *tensor_);
    }

    GGMLGraph(GGMLGraph&& other)
        : gf_(other.gf_), sched_(other.sched_), inputs_(std::move(other.inputs_)), tensor_(other.tensor_)
    {
        other.gf_ = nullptr;
    }

    ~GGMLGraph() {
        // TODO: should gf be freed?
    }

    ggml_tensor* node(int index) {
        return ggml_graph_node(gf_, index);
    }

    GGMLGraph(GGMLGraph&) = delete;
    GGMLGraph& operator =(const GGMLGraph&) = delete;

private:
    ggml_cgraph* gf_;
    ggml_backend_sched_t sched_;
    std::vector<Tensor> inputs_;
    Tensor tensor_;

    friend class GGMLScope;
};
