#pragma once

#include "ggml/Buffer.hpp"
#include "ggml/Allocator.hpp"
#include <ggml.h>
#include <vector>
#include <optional>

class Context;
class Device;
class DeviceAllocator : public Allocator {
public:
    DeviceAllocator(Context& context, const Device& device);

    /** @brief Allocates all unallocated tensors in the context from the device's buffer.
     *
     * May be called multiple times: each call allocates the tensors created since the
     * previous call. A call with nothing left to allocate is a no-op.
     */
    void allocate(const std::optional<ggml_backend_buffer_usage>& usage = std::nullopt);

    void reset();

    Context& context() {
        return context_;
    }

private:
    Context& context_;
    ggml_backend_buffer_type_t buft_;
    std::vector<Buffer> buffers_;
};
