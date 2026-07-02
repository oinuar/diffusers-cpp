#pragma once

#include "GGMLGraph.hpp"

#include "ggml.h"
#include "ggml-backend.h"

#include <vector>

class GGMLComputation {
public:
    GGMLComputation(GGMLGraph& graph) : graph_(graph) {
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
    std::pair<Tensor::Shape, std::vector<T>> read(int index) {
        auto tensor = ggml_graph_node(graph_.gf_, index);
        std::vector<T> data(ggml_nelements(tensor));

        // Bring the data from the backend memory
        ggml_backend_tensor_get(tensor, data.data(), 0, ggml_nbytes(tensor));

        Tensor::Shape shape(ggml_n_dims(tensor));

        for (auto i = 0; i < shape.rank(); ++i)
            shape[i] = tensor->ne[i];

        return {shape, std::move(data)};
    }

    GGMLComputation(GGMLComputation&) = delete;
    GGMLComputation(GGMLComputation&&) = delete;
    GGMLComputation& operator =(const GGMLComputation&) = delete;
    GGMLComputation& operator =(GGMLComputation&&) = delete;

private:
    GGMLGraph& graph_;
};
