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

    virtual void allocate(const std::vector<Context*>& contexts, const std::vector<Tensor>& outputs);

    virtual Runtime& runtime();

private:
    struct Usage {
        const Device* device;
        std::optional<ggml_backend_buffer_usage> usage;
        std::vector<Buffer> buffers;
    };

    std::unordered_map<Context*, Usage> usages_;
};
