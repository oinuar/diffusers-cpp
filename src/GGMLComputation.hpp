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

    void operator ()(const std::unordered_map<std::string, void*>& values) {
        // Validate values
        for (auto& pair : values) {
            auto it = graph_.params_.find(pair.first);

            if (it == std::end(graph_.params_))
                throw std::invalid_argument("No such parameter '" + pair.first + "'");
        }

        // Load data from CPU memory to backend buffer
        for (auto& pair : values) {
            auto& tensor = graph_.params_[pair.first];
            ggml_backend_tensor_set(*tensor, pair.second, 0, ggml_nbytes(*tensor));
        }

        // Perform the computation
        ggml_backend_sched_graph_compute(graph_.sched_, graph_.gf_);
    }

    template <class T>
    std::pair<std::array<int64_t, 4>, std::vector<T>> get(int index) {
        auto tensor = ggml_graph_node(graph_.gf_, index);
        std::vector<T> data(ggml_nelements(tensor));

        // Bring the data from the backend memory
        ggml_backend_tensor_get(tensor, data.data(), 0, ggml_nbytes(tensor));

        return {
            {tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]},
            std::move(data)
        };
    }

    GGMLComputation(GGMLComputation&) = delete;
    GGMLComputation(GGMLComputation&&) = delete;
    GGMLComputation& operator =(const GGMLComputation&) = delete;
    GGMLComputation& operator =(GGMLComputation&&) = delete;

private:
    GGMLGraph& graph_;
};
