#pragma once

#include "ggml/Buffer.hpp"
#include <ggml.h>
#include <unordered_map>
#include <optional>

class Allocator {
public:
    virtual void allocate(const std::optional<ggml_backend_buffer_usage>& usage = std::nullopt) = 0;
};
