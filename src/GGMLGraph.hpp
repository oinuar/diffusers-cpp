#pragma once

#include "Tensor.hpp"
#include "ggml.h"
#include "ggml-backend.h"

#include <vector>

class GGMLGraph {
public:
    GGMLGraph(ggml_cgraph* gf, ggml_backend_sched_t sched, std::unordered_map<std::string, Tensor>&& params, Tensor tensor)
        : gf_(gf), sched_(sched), params_(std::move(params)), tensor_(tensor)
    {
        ggml_build_forward_expand(gf_, *tensor_);
    }

    GGMLGraph(GGMLGraph&& other)
        : gf_(other.gf_), sched_(other.sched_), params_(std::move(other.params_)), tensor_(other.tensor_)
    {
        other.gf_ = nullptr;
    }

    GGMLGraph(GGMLGraph&) = delete;
    GGMLGraph& operator =(const GGMLGraph&) = delete;

private:
    ggml_cgraph* gf_;
    ggml_backend_sched_t sched_;
    std::unordered_map<std::string, Tensor> params_;
    Tensor tensor_;

    friend class GGMLComputation;
};
