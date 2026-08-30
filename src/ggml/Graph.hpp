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
          outputs_(std::move(outputs)),
          views_()
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

            // Ops that only change shape/stride (a view) without computing
            // data. A graph made up solely of these over a leaf has no compute
            // node, which the meta backend cannot place (it asserts in
            // split_graph when the last node is a skippable view). This matches
            // the no-compute set in ggml's backend supports_op.
            const auto is_noop_view = [](enum ggml_op op) {
                return op == GGML_OP_NONE || op == GGML_OP_RESHAPE || op == GGML_OP_VIEW ||
                       op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
            };

            // Walk the view chain down to the first tensor that computes
            // something or to the root leaf. Some ops (e.g. clamp) are
            // implemented as views with a compute op - those are real compute
            // nodes and must stay in the graph.
            auto pure_view_root = *tensor;

            for (auto t = pure_view_root; t->view_src != nullptr; t = t->view_src) {
                views_.push_back(t);

                if (is_noop_view(t->op))
                    pure_view_root = t->view_src;
            }

            // Pure view of a leaf: nothing to compute. Keep the leaf only so
            // the scheduler allocates it; the view nodes share its data.
            if (is_noop_view(pure_view_root->op)) {
                if (allocator == nullptr)
                    ggml_set_output(pure_view_root);

                ggml_build_forward_expand(gf_, pure_view_root);
                continue;
            }

            // Contains a compute op: keep the full graph so the compute nodes run.
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

    Runtime::Bindings allocate() {
        ggml_backend_sched_reset(*runtime_.scheduler());
        
        if (!ggml_backend_sched_alloc_graph(*runtime_.scheduler(), gf_))
            throw std::runtime_error("Graph allocation failed");

        // The allocator only initializes the buffer/data pointers of views it
        // can see in the graph. Views kept out of the graph (pure views of
        // leaves) need to be initialized here, from the leaf outwards.
        for (auto it = views_.rbegin(); it != views_.rend(); ++it) {
            if ((*it)->buffer == nullptr && (*it)->view_src->buffer != nullptr)
                ggml_backend_view_init(*it);
        }

        auto bindings = runtime_.bindings();

        for (auto it = std::begin(bindings); it != std::end(bindings); ) {
            // Remove tensors that have no bound buffer in this graph
            if ((*it->first)->buffer == nullptr) {
                it = bindings.erase(it);
                continue;
            }

            // Unbind tensors that are bound only once
            if (it->second.second)
                runtime_.unbind(it->first);

            ++it;
        }

        return std::move(bindings);
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
    std::vector<ggml_tensor*> views_;
};
