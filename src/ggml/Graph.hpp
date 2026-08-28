#pragma once

#include "ggml/Tensor.hpp"
#include "ggml/Runtime.hpp"
#include <ggml.h>
#include <ggml-backend.h>
#include <vector>
#include <random>
#include <iostream>

class Graph {
public:
    Graph(Runtime& runtime, std::vector<Tensor>&& outputs)
        : runtime_(runtime),
          gf_(ggml_new_graph_custom(*runtime.context(), runtime.scheduler().capacity(), false)),
          outputs_(std::move(outputs))
    {
        for (auto& tensor : outputs_) {
            // Materialize tensor if needed
            if (!tensor.is_contiguous())
                tensor = tensor.contiguous();

            ggml_set_output(*tensor);
            ggml_build_forward_expand(gf_, *tensor);
        }
    }

    Graph(Graph&& other) : runtime_(other.runtime_), gf_(other.gf_), outputs_(std::move(other.outputs_)) {
        other.gf_ = nullptr;
    }

    ggml_cgraph* operator *() {
        return gf_;
    }

    Runtime& runtime() {
        return runtime_;
    }

    std::vector<Tensor>& outputs() {
        return outputs_;
    }

    Graph(Graph&) = delete;
    Graph& operator =(const Graph&) = delete;
    Graph& operator =(Graph&&) = delete;

private:
    Runtime& runtime_;
    ggml_cgraph* gf_;
    std::vector<Tensor> outputs_;
};
