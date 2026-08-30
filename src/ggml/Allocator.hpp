#pragma once

#include "ggml/Buffer.hpp"
#include <ggml.h>
#include <unordered_map>
#include <optional>

class Tensor;
class Context;
class Device;

class Allocator {
public:
    Allocator(Context& context, const Device& device);

    void allocate(const std::optional<ggml_backend_buffer_usage>& usage = std::nullopt);

private:
    Context& context_;
    ggml_backend_buffer_type_t buft_;
    std::optional<Buffer> buffer_;
};
