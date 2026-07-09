#pragma once

#include "ggml/Tensor.hpp"
#include "ggml.h"
#include "ggml-backend.h"

#include <vector>

class Graph {
public:
    Graph(ggml_cgraph* gf, ggml_backend_sched_t sched, std::vector<Tensor>&& tensors)
        : gf_(gf), sched_(sched), tensors_(tensors)
    {
        for (auto& tensor : tensors_)
            ggml_build_forward_expand(gf_, *tensor);
    }

    Graph(Graph&& other)
        : gf_(other.gf_), sched_(other.sched_), tensors_(std::move(other.tensors_))
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
    std::vector<Tensor> tensors_;

    friend class Computation;
};
