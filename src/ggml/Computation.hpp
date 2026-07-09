#pragma once

#include "Graph.hpp"

#include "ggml.h"
#include "ggml-backend.h"

#include <vector>

class Computation {
public:
    template <typename T> 
    using Result = std::vector<std::pair<Tensor::Shape, std::vector<T>>>;

    Computation(Graph& graph) : graph_(graph) {
        ggml_backend_sched_reset(graph.sched_);
        ggml_backend_sched_alloc_graph(graph.sched_, graph.gf_);
    }

    void execute() {
        ggml_backend_sched_graph_compute(graph_.sched_, graph_.gf_);
    }

    void load(Tensor& tensor, std::byte* values) {
        ggml_backend_tensor_set(*tensor, values, 0, ggml_nbytes(*tensor));
    }

    template <class T>
    Result<T> read() {
        Result<T> result;

        for (auto& tensor : graph_.tensors_) {
            std::vector<T> data(ggml_nelements(*tensor));

            // Bring the data from the backend memory
            ggml_backend_tensor_get(*tensor, data.data(), 0, ggml_nbytes(*tensor));

            result.push_back({tensor.shape(), std::move(data)});
        }

        return std::move(result);
    }

    Computation(Computation&) = delete;
    Computation(Computation&&) = delete;
    Computation& operator =(const Computation&) = delete;
    Computation& operator =(Computation&&) = delete;

private:
    Graph& graph_;
};
