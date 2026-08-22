#pragma once

#include "Context.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include <vector>

class Graph;

class Scheduler {
public:
    Scheduler(std::vector<ggml_backend_t>&& backends, size_t graph_size = GGML_DEFAULT_GRAPH_SIZE)
        : backends_(std::move(backends)), sched_(nullptr), allocated_graph_(nullptr), graph_size_(graph_size)
    {
        sched_ = ggml_backend_sched_new(backends_.data(), nullptr, backends_.size(), graph_size, false, true);
    }

    Scheduler(Scheduler&& other)
        : backends_(std::move(other.backends_)), sched_(other.sched_), allocated_graph_(other.allocated_graph_), graph_size_(other.graph_size_)
    {
        other.sched_ = nullptr;
        other.allocated_graph_ = nullptr;
    }

    ~Scheduler() {
        if (sched_ != nullptr)
            ggml_backend_sched_free(sched_);
    }

    ggml_backend_sched_t operator *() { return sched_; }

    size_t capacity() const {
        return graph_size_;
    }

    void allocate(ggml_cgraph* graph, ggml_backend_sched_eval_callback callback, void* user_data) {
        // Nothing to do if graph is already allocated in scheduler
        if (graph == allocated_graph_)
            return;
        
        ggml_backend_sched_reset(sched_);
        ggml_backend_sched_alloc_graph(sched_, graph);

        if (callback != nullptr)
            ggml_backend_sched_set_eval_callback(sched_, callback, user_data);

        allocated_graph_ = graph;
    }

    Scheduler(Scheduler&) = delete;
    Scheduler& operator =(const Scheduler&) = delete;

private:
    std::vector<ggml_backend_t> backends_;
    ggml_backend_sched_t sched_;
    ggml_cgraph* allocated_graph_;
    size_t graph_size_;
};
