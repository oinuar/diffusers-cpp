#pragma once

#include "ggml/Backend.hpp"
#include <vector>

class Graph;

class Scheduler {
public:
    Scheduler(std::vector<Backend*>&& backends, size_t graph_size = GGML_DEFAULT_GRAPH_SIZE)
        : backends_(std::move(backends)), sched_(nullptr), graph_size_(graph_size)
    {
        std::vector<ggml_backend_t> ggml_backends;

        for (auto backend : backends_)
            ggml_backends.push_back(**backend);

        sched_ = ggml_backend_sched_new(ggml_backends.data(), nullptr, ggml_backends.size(), graph_size, false, false);
    }

    Scheduler(Scheduler&& other)
        : backends_(std::move(other.backends_)), sched_(other.sched_), graph_size_(other.graph_size_)
    {
        other.sched_ = nullptr;
    }

    ~Scheduler() {
        if (sched_ != nullptr)
            ggml_backend_sched_free(sched_);
    }

    ggml_backend_sched_t operator *() { return sched_; }

    size_t capacity() const {
        return graph_size_;
    }

    std::vector<Backend*>& backends() {
        return backends_;
    }

    Scheduler(Scheduler&) = delete;
    Scheduler& operator =(const Scheduler&) = delete;

private:
    std::vector<Backend*> backends_;
    ggml_backend_sched_t sched_;
    size_t graph_size_;
};
