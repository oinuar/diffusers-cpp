#include "ggml/Allocator.hpp"
#include "ggml/Context.hpp"
#include "ggml/Device.hpp"
#include "ggml/ExecutionEngine.hpp"

Allocator::Allocator() : usages_(), buffers_() {

}

void Allocator::use(Context& context, const Device& device, ggml_backend_buffer_usage usage) {
    usages_.push_back({&context, &device, usage});
}

void Allocator::unuse(const Context& context) {
    usages_.erase(std::remove_if(usages_.begin(), usages_.end(),
        [&](const Usage& usage) { return usage.context == &context; }), usages_.end());
}

void Allocator::allocate(const std::vector<Tensor>&, bool reallocate) {
    if (usages_.empty() || reallocate)
        buffers_.clear();

    for (auto& [context, device, usage] : usages_) {
        auto buft = device->buffer_type();
        auto buffer = ggml_backend_alloc_ctx_tensors_from_buft(**context, buft);

        // NULL is returned when every tensor in the context already has a buffer,
        // e.g. when the graph is built entirely in another context. There is
        // nothing to allocate in that case.
        if (buffer == nullptr)
            return;

        buffers_.emplace_back(buffer, usage);
    }
}

void Allocator::reset() {
    usages_.clear();
    buffers_.clear();
}

Engine& Allocator::engine() {
    return ExecutionEngine::Default;
}
