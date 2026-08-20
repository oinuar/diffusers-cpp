#pragma once

#include "ggml/Runtime.hpp"
#include "ggml/Scheduler.hpp"
#include "ggml/Graph.hpp"
#include "ProgressBar.hpp"
#include <ggml.h>
#include <ggml-backend.h>
#include <vector>

class Computation {
public:
    explicit Computation(Graph& graph, ProgressBar* progress = nullptr) : graph_(graph) {
        ggml_backend_sched_reset(*graph.scheduler());
        ggml_backend_sched_alloc_graph(*graph.scheduler(), *graph);

        auto inputs = graph.inputs();

        if (progress != nullptr)
            progress->push("Initializing", inputs.size() + 1);

        for (auto& [tensor, provider] : inputs) {
            load(tensor, graph.provide(provider));

            if (progress != nullptr)
                progress->next();
        }

        if (progress != nullptr) {
            progress->pop();
            progress->push("Computing", 1);
        }

        ggml_backend_sched_graph_compute(*graph.scheduler(), *graph);

        if (progress != nullptr) {
            progress->next();
            progress->pop();
        }
    }

    const std::vector<Tensor>& results() const {
        return graph_.tensors();
    }

    Computation(Computation&) = delete;
    Computation(Computation&&) = delete;
    Computation& operator =(const Computation&) = delete;
    Computation& operator =(Computation&&) = delete;

private:
    Graph& graph_;

    void load(const Tensor& tensor, const std::vector<std::byte>& bytes) {
        if (bytes.size() != ggml_nbytes(*tensor))
            throw std::invalid_argument("load(): data size mismatch: expected " + std::to_string(bytes.size()) + ", but got " + std::to_string(ggml_nbytes(*tensor)));

        ggml_backend_tensor_set(
            *tensor,
            bytes.data(),
            0,
            ggml_nbytes(*tensor)
        );
    }
};
