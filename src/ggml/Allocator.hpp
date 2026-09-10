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
    Allocator();

    virtual ~Allocator() = default;

    void use(Context& context, const Device& device, const std::optional<ggml_backend_buffer_usage>& usage = std::nullopt);

    void unuse(const Context& context);

    virtual void allocate(const std::vector<Tensor>& outputs, bool reallocate = false);

    virtual void reset();

    virtual Runtime& runtime();

private:
    struct Usage {
        Context* context;
        const Device* device;
        std::optional<ggml_backend_buffer_usage> usage;
    };

    std::vector<Usage> usages_;
    std::vector<Buffer> buffers_;
};
