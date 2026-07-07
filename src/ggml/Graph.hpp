#pragma once

#include "ggml/Tensor.hpp"
#include "ggml.h"
#include "ggml-backend.h"

#include <vector>

class Graph {
public:
    Graph(ggml_cgraph* gf, ggml_backend_sched_t sched, Tensor tensor)
        : gf_(gf), sched_(sched), tensor_(tensor)
    {
        ggml_build_forward_expand(gf_, *tensor_);
    }

    Graph(Graph&& other)
        : gf_(other.gf_), sched_(other.sched_), tensor_(other.tensor_)
    {
        other.gf_ = nullptr;
    }

    ggml_cgraph* operator *() {
        return gf_;
    }

    Graph(Graph&) = delete;
    Graph& operator =(const Graph&) = delete;

private:
    ggml_cgraph* gf_;
    ggml_backend_sched_t sched_;
    Tensor tensor_;

    friend class Computation;
};
