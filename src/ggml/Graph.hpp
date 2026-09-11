#pragma once

#include "ggml/Tensor.hpp"
#include "ggml/Context.hpp"
#include "ggml/Scope.hpp"
#include "ggml/Scheduler.hpp"
#include <ggml.h>
#include <ggml-backend.h>
#include <vector>
#include <random>
#include <iostream>

class Graph {
public:
    Graph(Scheduler& scheduler, Context& context, std::vector<Tensor>&& outputs, size_t capacity = GGML_DEFAULT_GRAPH_SIZE)
        : scheduler_(scheduler), context_(context), gf_(ggml_new_graph_custom(*context_, context_.capacity(), false)), outputs_(std::move(outputs))
    {
        Scope scope(context_);

        for (auto& tensor : outputs_) {
            // The meta backend skips a node whose view_src is a static
            // tensor (a GGML_OP_NONE in a host buffer) and asserts that the
            // graph's last node is not such a skip (ggml_backend_meta
            // _graph_compute: i_start == n_nodes). ggml flattens view chains
            // in ggml_set_view_op, so view_src is always the base tensor. A graph
            // that ends in a view of a static tensor has no computable last node,
            // so materialize it with a copy: the dup is a compute node the meta 
            // can place, and it keeps the output in the same (replicated) distribution
            // the plan committed.
            if ((*tensor)->view_src != nullptr && (*tensor)->view_src->op == GGML_OP_NONE)
                tensor = tensor.clone();

            // Materialize output tensor by making it contiguous if needed.
            else if (!tensor.is_contiguous())
                tensor = tensor.contiguous();

            // Set output & build the graph.
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

        // In a sharding setup (a meta backend in the scheduler), run the whole graph through
        // the meta backend (no-op otherwise). See Scheduler::meta_backend for why the meta
        // backend must be the runtime of the entire graph. Pinning is a user assignment,
        // which the scheduler honors in split_graph; it must happen after the reset above
        // (a reset wipes user assignments) and before ggml_backend_sched_alloc_graph below.
        // The graph's tensors live in the Graph context (inputs and compute nodes) and the
        // additional contexts (e.g. weights), so pin every tensor of those contexts.
        if (ggml_backend_t meta = scheduler_.meta_backend()) {
            auto pin_context = [&](Context& context) {
                for (ggml_tensor* tensor = ggml_get_first_tensor(*context); tensor != nullptr; tensor = ggml_get_next_tensor(*context, tensor))
                    ggml_backend_sched_set_tensor_backend(*scheduler_, tensor, meta);
            };

            pin_context(context_);

            for (auto& context : contexts)
                if (context != nullptr && context != &context_)
                    pin_context(*context);
        }

        if (!ggml_backend_sched_alloc_graph(*scheduler_, gf_))
            throw std::runtime_error("Graph allocation failed");
        Context::Bindings result;

        // Add Graph context bindings
        for (auto& [tensor, binding] : context_.bindings()) {
            if (!binding.unbound && (*tensor)->buffer != nullptr)
                result.insert(std::make_pair(tensor, binding));
        }

        // Unbind one-time bound tensors from Graph's context
        for (auto& [tensor, binding] : result) {
            if (binding.once)
                context_.unbind(tensor);
        }

        for (auto& context : contexts) {
            // Skip null context
            if (context == nullptr)
                continue;

            // Skip Graph's context
            if (context == &context_)
                continue;

            Context::Bindings bindings;

            // Collect additional context bindings
            for (auto& [tensor, binding] : context->bindings()) {
                if (!binding.unbound && (*tensor)->buffer != nullptr)
                    bindings.insert(std::make_pair(tensor, binding));
            }

            // Unbind tensors that are bound only once
            for (auto& [tensor, binding] : bindings) {
                if (binding.once)
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
    Scheduler& scheduler_;
    Context& context_;
    ggml_cgraph* gf_;
    std::vector<Tensor> outputs_;
};
