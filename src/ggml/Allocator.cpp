#include "ggml/Allocator.hpp"
#include "ggml/Context.hpp"
#include "ggml/Device.hpp"
#include "ggml/ExecutionRuntime.hpp"

Allocator::Allocator() : usages_() {

}

void Allocator::use(Context& context, const Device& device, const std::optional<ggml_backend_buffer_usage>& usage) {
    usages_[&context] = {&device, usage, {}};
}

void Allocator::allocate(const std::vector<Context*>& contexts, const std::vector<Tensor>&) {
    for (auto& context : contexts) {
        auto it = usages_.find(context);

        // Skip allocating context if it is not known by this Allocator.
        if (it == std::end(usages_))
            continue;

        auto buft = it->second.device->buffer_type();
        auto buffer = ggml_backend_alloc_ctx_tensors_from_buft(**context, buft);

        // NULL is returned when every tensor in the context already has a buffer,
        // e.g. when the graph is built entirely in another context. There is
        // nothing to allocate in that case.
        if (buffer == nullptr)
            continue;

        it->second.buffers.emplace_back(buffer, it->second.usage);
    }
}

Runtime& Allocator::runtime() {
    return ExecutionRuntime::Default;
}
