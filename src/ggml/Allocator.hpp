#pragma once

#include "ggml/Tensor.hpp"
#include "ggml/Buffer.hpp"
#include <ggml-backend.h>
#include <optional>
#include <unordered_map>

class Context;
class Device;
class Runtime;

class Allocator {
public:
    Allocator(Device& device, const std::optional<ggml_backend_buffer_usage>& usage = std::nullopt);

    virtual ~Allocator() = default;

    virtual void allocate(Context& context);

    virtual void deallocate(Context& context);

    const Device& device() const {
        return device_;
    }

protected:
    static std::string format_bytes(size_t bytes);

private:
    Device& device_;
    std::optional<ggml_backend_buffer_usage> usage_;
    std::unordered_multimap<ggml_context*, Buffer> buffers_;
};
