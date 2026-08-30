#pragma once

#include "ggml/Tensor.hpp"
#include "ggml/Runtime.hpp"
#include "ggml/Allocator.hpp"
#include <ggml.h>
#include <ggml-backend.h>
#include <vector>
#include <random>
#include <iostream>

class Graph {
public:
    Graph(Runtime& runtime, std::vector<Tensor>&& outputs, Allocator* allocator = nullptr)
        : runtime_(runtime),
          gf_(ggml_new_graph_custom(*runtime.context(), runtime.scheduler().capacity(), false)),
          outputs_(std::move(outputs))
    {
        for (auto& tensor : outputs_) {
            // Relocate tensors to given allocator if required
            if (allocator != nullptr && !allocator->contains(tensor)) {
                auto leaf = Tensor::empty(*runtime.context(), tensor.shape(), tensor.dtype(), allocator);

                ggml_build_forward_expand(gf_, *tensor.copy_to(leaf));

                tensor = leaf;
                continue;
            }

            // Materialize tensor if needed
            if (!tensor.is_contiguous())
                tensor = tensor.contiguous();

            // Set output flag if allocator is not given to
            // keep tensor allocated by the scheduler
            if (allocator == nullptr)
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
    std::vector<std::pair<Tensor, Tensor>> copies_;
};
