#pragma once

#include "ggml/Tensor.hpp"
#include "ggml/Buffer.hpp"
#include <ggml-backend.h>
#include <optional>

class Context;
class Device;
class Runtime;

class Allocator {
public:
    Allocator(Device& device, const std::optional<ggml_backend_buffer_usage>& usage = std::nullopt);

    virtual ~Allocator() = default;

    virtual void allocate(Context& context);

private:
    Device& device_;
    std::optional<ggml_backend_buffer_usage> usage_;
    std::vector<Buffer> buffers_;
};
