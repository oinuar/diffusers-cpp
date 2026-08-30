#pragma once

#include "ggml/Context.hpp"
#include "ggml/Scheduler.hpp"
#include "ggml/Graph.hpp"
#include "ProgressBar.hpp"
#include <ggml.h>
#include <cassert>
#include <random>

class Computation {
public:
    explicit Computation(Graph& graph, std::initializer_list<Context*>&& contexts = {}, ProgressBar* progress = nullptr, uint64_t seed = std::random_device{}())
        : graph_(graph), progress_(progress), bindings_(graph.context().bindings()), rng_(seed), computed_(false)
    {
        graph_.allocate();

        // Unbind one-time bound tensors from graph's context
        for (auto& [tensor, binding] : bindings_) {
            if (binding.second)
                graph.context().unbind(tensor);
        }

        for (auto& context : contexts) {
            // Skip graph's context
            if (context == &graph.context())
                continue;

            auto bindings = context->bindings();

            // Unbind tensors that are bound only once
            for (auto& [tensor, binding] : bindings) {
                if (binding.second)
                    context->unbind(tensor);
            }

            // Add additional context bindings
            bindings_.insert(std::begin(bindings), std::end(bindings));
        }
    }
    
    ~Computation() {
        assert(computed_ && "Computation was never run. Don't create it if you don't need it.");
    }

    Computation& operator ()() {
        if (progress_ != nullptr)
            progress_->push("Initializing", bindings_.size() + 1);

        for (auto& [tensor, binding] : bindings_) {
            graph_.context().write(tensor, binding.first(rng_));

            if (progress_ != nullptr)
                progress_->next();
        }

        if (progress_ != nullptr) {
            progress_->pop();
            progress_->push("Computing", 1);
        }

        ggml_backend_sched_graph_compute(*graph_.scheduler(), *graph_);

        if (progress_ != nullptr) {
            progress_->next();
            progress_->pop();  
        }

        computed_ = true;
        return *this;
    }

    const std::vector<Tensor>& results() const {
        if (!computed_)
            throw std::runtime_error("Access of undefined results: computation was not run");

        return graph_.outputs();
    }

    Computation(Computation&) = delete;
    Computation(Computation&&) = delete;
    Computation& operator =(const Computation&) = delete;
    Computation& operator =(Computation&&) = delete;

private:
    Graph& graph_;
    ProgressBar* progress_;
    Context::Bindings bindings_;
    std::mt19937 rng_;
    bool computed_;
};
