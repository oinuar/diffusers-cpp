#pragma once

#include "ggml/Tensor.hpp"
#include "ggml/Runtime.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include <vector>
#include <random>

class Graph {
public:
    Graph(Runtime& runtime, std::vector<Tensor>&& tensors)
        : runtime_(runtime), gf_(ggml_new_graph(*runtime.context())), tensors_(std::move(tensors))
    {
        for (auto& tensor : tensors_) {
            // Materialize tensor if needed
            if (!tensor.is_contiguous())
                tensor = tensor.contiguous();

            ggml_set_output(*tensor);
            ggml_build_forward_expand(gf_, *tensor);
        }
    }

    ggml_cgraph* operator *() {
        return gf_;
    }

    std::vector<std::byte> provide(const Runtime::Provider<std::byte>& provider) {
        return std::move(provider(runtime_.rng_));
    }

    Runtime::Inputs inputs() {
        return runtime_.inputs_;
    }

    std::vector<Tensor>& tensors() {
        return tensors_;
    }

    Scheduler& scheduler() {
        return runtime_.scheduler_;
    }

    Graph(Graph&) = delete;
    Graph(Graph&&) = default;
    Graph& operator =(const Graph&) = delete;
    Graph& operator =(Graph&&) = default;

private:
    Runtime& runtime_;
    ggml_cgraph* gf_;
    std::vector<Tensor> tensors_;
};
