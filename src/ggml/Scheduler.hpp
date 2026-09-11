#pragma once

#include "ggml/Backend.hpp"
#include <vector>

class Graph;

class Scheduler {
public:
    Scheduler(std::vector<Backend*>&& backends, size_t capacity = GGML_DEFAULT_GRAPH_SIZE)
        : backends_(std::move(backends)), sched_(nullptr)
    {
        std::vector<ggml_backend_t> ggml_backends;

        for (auto backend : backends_)
            ggml_backends.push_back(**backend);

        sched_ = ggml_backend_sched_new(ggml_backends.data(), nullptr, ggml_backends.size(), capacity, false, true);
    }

    Scheduler(Scheduler&& other)
        : backends_(std::move(other.backends_)), sched_(other.sched_)
    {
        other.sched_ = nullptr;
    }

    ~Scheduler() {
        if (sched_ != nullptr)
            ggml_backend_sched_free(sched_);
    }

    ggml_backend_sched_t operator *() {
        return sched_;
    }

    std::vector<Backend*>& backends() {
        return backends_;
    }

    /**
     * @brief Returns the meta (sharding) backend of the scheduler, or nullptr if there is none.
     *
     * In a sharding setup the meta backend is the only runtime that can read tensors in a meta
     * buffer: a meta buffer exposes a fake base, the real data lives in the per-device simple
     * tensors, and the meta backend computes per-device subgraphs on them. The scheduler's own
     * placement heuristics cannot express this, because the meta backend does not support the
     * fallback backends' buffer types (and those do not support the meta ones without the meta
     * backend in between), so they end up splitting the graph onto the fallback backend, which
     * would dereference the fake pointers. When a meta backend is present it is the runtime for
     * the whole sharded graph; the fallback backends are only used through the meta backend's
     * simple devices, so every tensor of the graph must be pinned to it before allocation.
     */
    ggml_backend_t meta_backend() const {
        for (auto& backend : backends_)
            if (ggml_backend_dev_type(*backend->device()) == GGML_BACKEND_DEVICE_TYPE_META)
                return **backend;

        return nullptr;
    }

    Scheduler(Scheduler&) = delete;
    Scheduler& operator =(const Scheduler&) = delete;

private:
    std::vector<Backend*> backends_;
    ggml_backend_sched_t sched_;
};
