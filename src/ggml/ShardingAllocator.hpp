#pragma once

#include "ggml/Allocator.hpp"
#include "ggml/ShardingRuntime.hpp"

class ShardingAllocator : public Allocator {
public:
    ShardingAllocator(ShardingRuntime& runtime, MetaDevice& device, const std::optional<ggml_backend_buffer_usage>& usage = std::nullopt)
        : Allocator(device, usage), runtime_(runtime) {}

    virtual ~ShardingAllocator() = default;

    void allocate(Context& context) override;

    const ShardingRuntime& runtime() const { return runtime_; }

private:
    ShardingRuntime& runtime_;
};
