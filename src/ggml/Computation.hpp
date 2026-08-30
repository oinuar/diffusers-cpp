#pragma once

#include "ggml/Runtime.hpp"
#include "ggml/Scheduler.hpp"
#include "ggml/Graph.hpp"
#include "ProgressBar.hpp"
#include <ggml.h>
#include <cassert>

class Computation {
public:
    explicit Computation(Graph& graph, ProgressBar* progress = nullptr)
        : graph_(graph), progress_(progress), computed_(false), bindings_(std::move(graph.allocate()))
    {
    }
    
    ~Computation() {
        assert(computed_ && "Computation was never run. Don't create it if you don't need it.");
    }

    Computation& operator ()() {
        if (progress_ != nullptr)
            progress_->push("Initializing", bindings_.size() + 1);

        for (auto& [tensor, value] : bindings_) {
            graph_.runtime().write(tensor, value.first(graph_.runtime().rng()));

            if (progress_ != nullptr)
                progress_->next();
        }

        if (progress_ != nullptr) {
            progress_->pop();
            progress_->push("Computing", 1);
        }

        ggml_backend_sched_graph_compute(*graph_.runtime().scheduler(), *graph_);

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
    Runtime::Bindings bindings_;
    bool computed_;
};
