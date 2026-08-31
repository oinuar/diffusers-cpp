#pragma once

#include "ggml/Buffer.hpp"
#include "ggml/Allocator.hpp"
#include <ggml.h>
#include <unordered_map>
#include <optional>

class Context;
class Device;

class DeviceAllocator : public Allocator {
public:
    DeviceAllocator(Context& context, const Device& device);

    void allocate(const std::optional<ggml_backend_buffer_usage>& usage = std::nullopt);

    Context& context() {
        return context_;
    }

    const Buffer& buffer() const {
        return *buffer_;
    }

private:
    Context& context_;
    ggml_backend_buffer_type_t buft_;
    std::optional<Buffer> buffer_;
};
