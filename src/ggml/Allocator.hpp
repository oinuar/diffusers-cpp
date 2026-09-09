#pragma once

#include "ggml/Tensor.hpp"
#include "ggml/Buffer.hpp"
#include <ggml-backend.h>

class Context;
class Device;
class Engine;

class Allocator {
public:
    Allocator();

    virtual ~Allocator() = default;

    void use(Context& context, const Device& device, ggml_backend_buffer_usage usage);

    void unuse(const Context& context);

    virtual void allocate(const std::vector<Tensor>& outputs, bool reallocate = false);

    virtual void reset();

    virtual Engine& engine();

private:
    struct Usage {
        Context* context;
        const Device* device;
        ggml_backend_buffer_usage usage;
    };

    std::vector<Usage> usages_;
    std::vector<Buffer> buffers_;
};
