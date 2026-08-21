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
    Graph(Runtime& runtime, std::vector<Tensor>&& tensors)
        : runtime_(runtime),
          gf_(ggml_new_graph_custom(*runtime.context(), runtime.scheduler_.capacity(), false)),
          tensors_(std::move(tensors)),
          is_computing_(false)
    {
        for (auto& tensor : tensors_) {
            // Materialize tensor if needed
            if (!tensor.is_contiguous())
                tensor = tensor.contiguous();

            ggml_set_output(*tensor);
            ggml_build_forward_expand(gf_, *tensor);
        }
    }

    Graph(Graph&& other) : runtime_(other.runtime_), gf_(other.gf_), tensors_(std::move(other.tensors_)) {
        other.gf_ = nullptr;
    }

    ggml_cgraph* operator *() {
        return gf_;
    }

    std::vector<std::byte> provide(const Runtime::Provider<std::byte>& provider) {
        return std::move(provider(runtime_.rng_));
    }

    Runtime::Inputs inputs() const {
        auto inputs = runtime_.inputs_;

        // Remove tensors from input that have no bound buffer. This can happen when
        // for example there are bound values in Runtime for tensors that are not used
        // by this graph. So we can ignore such tensors here.
        for (auto it = std::begin(inputs); it != std::end(inputs); ) {
            if ((*it->first)->buffer == nullptr)
                it = inputs.erase(it);
            else
                ++it;
        }

        return std::move(inputs);
    }

    std::vector<Tensor>& tensors() {
        return tensors_;
    }

    Graph(Graph&) = delete;
    Graph& operator =(const Graph&) = delete;
    Graph& operator =(Graph&&) = delete;
private:
    Runtime& runtime_;
    ggml_cgraph* gf_;
    std::vector<Tensor> tensors_;
    bool is_computing_;

    Scheduler& scheduler() {
        return runtime_.scheduler_;
    }

    friend class Computation;
};
