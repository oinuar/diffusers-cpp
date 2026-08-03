#pragma once

#include "Arena.hpp"
#include "Context.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include <vector>

class Scheduler {
public:
    Scheduler(std::vector<ggml_backend_t>&& backends, size_t graph_size = GGML_DEFAULT_GRAPH_SIZE)
        : backends_(std::move(backends)), arena_(graph_size)
    {
        sched_ = ggml_backend_sched_new(backends_.data(), nullptr, backends_.size(), graph_size, false, true);
    }

    Scheduler(Scheduler&& other)
        : backends_(std::move(other.backends_)), arena_(std::move(other.arena_)), sched_(other.sched_)
    {
        other.sched_ = nullptr;
    }

    ~Scheduler() {
        if (sched_ != nullptr)
            ggml_backend_sched_free(sched_);
    }

    Arena& arena() { return arena_; }

    ggml_backend_sched_t operator *() { return sched_; }

    Scheduler(Scheduler&) = delete;
    Scheduler& operator =(const Scheduler&) = delete;
private:
    std::vector<ggml_backend_t> backends_;
    Arena arena_;
    ggml_backend_sched_t sched_;
};
