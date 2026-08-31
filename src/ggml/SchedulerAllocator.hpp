#pragma once

#include "ggml/Allocator.hpp"

class SchedulerAllocator : public Allocator {
public:
    virtual void allocate(const std::optional<ggml_backend_buffer_usage>& = std::nullopt) {}
};
