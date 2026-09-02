#pragma once

#include "ggml/Tensor.hpp"
#include "ggml/Context.hpp"
#include "ggml/Scheduler.hpp"
#include <ggml.h>
#include <ggml-backend.h>
#include <vector>
#include <random>
#include <iostream>

class Graph {
public:
    Graph(Scheduler& scheduler, Context& context, std::vector<Tensor>&& outputs, size_t capacity = GGML_DEFAULT_GRAPH_SIZE)
        : scheduler_(scheduler), context_(context), gf_(ggml_new_graph_custom(*context_, context_.capacity(), false)), outputs_(std::move(outputs)), views_()
    {
        for (auto& tensor : outputs_) {
            // Materialize tensor if needed.
            if (!tensor.is_contiguous()) {
                Scope scope(context_);
                tensor = tensor.contiguous();
            }

            auto view_only = *tensor;
            auto is_pure_view = true;

            // Walk the view chain down to the first tensor that computes
            // something or to the root leaf.
            for (auto t = *tensor; t->view_src != nullptr; t = t->view_src) {
                views_.push_back(t);

                if (is_noop_view(t->op)) 
                    view_only = t->view_src;
                else
                    is_pure_view = false;
            }

            // Pure view of a leaf: nothing to compute, keep the leaf only if it has the backing buffer.
            if (is_noop_view(view_only->op) && is_pure_view) {
                ggml_set_output(view_only);
                ggml_build_forward_expand(gf_, view_only);
                continue;
            }

            views_.clear();

            // Otherwise, this is a real compute node.
            ggml_set_output(*tensor);
            ggml_build_forward_expand(gf_, *tensor);
        }
    }

    Graph(Graph&& other)
        : scheduler_(other.scheduler_), context_(other.context_), gf_(other.gf_), outputs_(std::move(other.outputs_))
    {
        other.gf_ = nullptr;
    }

    Context::Bindings allocate(std::initializer_list<Context*>&& contexts = {}) {
        ggml_backend_sched_reset(*scheduler_);
        
        if (!ggml_backend_sched_alloc_graph(*scheduler_, gf_))
            throw std::runtime_error("Graph allocation failed");

        // The allocator only initializes the buffer/data pointers of views it
        // can see in the graph. Views kept out of the graph (pure views of
        // leaves) need to be initialized here, from the leaf outwards.
        for (auto it = views_.rbegin(); it != views_.rend(); ++it) {
            if ((*it)->buffer == nullptr && (*it)->view_src->buffer != nullptr)
                ggml_backend_view_init(*it);
        }

        Context::Bindings result;

        // Add Graph context bindings
        for (auto& [tensor, binding] : context_.bindings()) {
            if ((*tensor)->buffer != nullptr)
                result.insert(std::make_pair(tensor, binding));
        }

        // Unbind one-time bound tensors from Graph's context
        for (auto& [tensor, binding] : result) {
            if (binding.second)
                context_.unbind(tensor);
        }

        for (auto& context : contexts) {
            // Skip Graph's context
            if (context == &context_)
                continue;

            Context::Bindings bindings;

            // Collect additional context bindings
            for (auto& [tensor, binding] : context->bindings()) {
                if ((*tensor)->buffer != nullptr)
                    bindings.insert(std::make_pair(tensor, binding));
            }

            // Unbind tensors that are bound only once
            for (auto& [tensor, binding] : bindings) {
                if (binding.second)
                    context->unbind(tensor);
            }

            // Add additional context bindings
            result.insert(std::begin(bindings), std::end(bindings));
        }

        return std::move(result);
    }

    ggml_cgraph* operator *() {
        return gf_;
    }

    Context& context() {
        return context_;
    }

    Scheduler& scheduler() {
        return scheduler_;
    }

    std::vector<Tensor>& outputs() {
        return outputs_;
    }

    Graph(Graph&) = delete;
    Graph& operator =(const Graph&) = delete;
    Graph& operator =(Graph&&) = delete;

private:

    // Ops that only change shape/stride (a view) without computing
    // data. A graph made up solely of these over a leaf has no compute
    // node, which the meta backend cannot place (it asserts in
    // split_graph when the last node is a skippable view). This matches
    // the no-compute set in ggml's backend supports_op.
    static bool is_noop_view(enum ggml_op op) {
        return op == GGML_OP_NONE || op == GGML_OP_RESHAPE || op == GGML_OP_VIEW ||
                op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
    };

    Scheduler& scheduler_;
    Context& context_;
    ggml_cgraph* gf_;
    std::vector<Tensor> outputs_;
    std::vector<ggml_tensor*> views_;
};
