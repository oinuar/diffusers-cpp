#pragma once

#include "Tensor.hpp"
#include "ggml.h"
#include "ggml-backend.h"

#include <vector>

class GGMLGraph {
public:
    GGMLGraph(ggml_cgraph* gf, ggml_backend_sched_t sched, Tensor tensor)
        : gf_(gf), sched_(sched), tensor_(tensor)
    {
        ggml_build_forward_expand(gf_, *tensor_);
    }

    GGMLGraph(GGMLGraph&& other)
        : gf_(other.gf_), sched_(other.sched_), tensor_(other.tensor_)
    {
        other.gf_ = nullptr;
    }

    ggml_cgraph* operator *() {
        return gf_;
    }

    GGMLGraph(GGMLGraph&) = delete;
    GGMLGraph& operator =(const GGMLGraph&) = delete;

private:
    ggml_cgraph* gf_;
    ggml_backend_sched_t sched_;
    Tensor tensor_;

    friend class GGMLComputation;
};
