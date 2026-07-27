#pragma once

#include "ggml/Tensor.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include <vector>
#include <random>

class Runtime;

class Graph {
public:
    using Initializers = std::vector<std::pair<Tensor, std::function<std::vector<std::byte>(Tensor, std::mt19937&)>>>;
    
    Graph(ggml_cgraph* gf, ggml_backend_sched_t sched, std::mt19937&& rng, Initializers&& initializers, std::vector<Tensor>&& tensors)
        : gf_(gf), sched_(sched), rng_(rng), initializers_(initializers), tensors_(tensors)
    {
        for (auto& tensor : tensors_) {
            // Materialize tensor if needed
            if (!tensor.is_contiguous())
                tensor = tensor.contiguous();

            ggml_set_output(*tensor);
            ggml_build_forward_expand(gf_, *tensor);
        }
    }

    Graph(Graph&& other)
        : gf_(other.gf_), sched_(other.sched_), rng_(std::move(other.rng_)), initializers_(std::move(other.initializers_)), tensors_(std::move(other.tensors_))
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
    std::mt19937 rng_;
    Initializers initializers_;
    std::vector<Tensor> tensors_;

    friend class Computation;
};
