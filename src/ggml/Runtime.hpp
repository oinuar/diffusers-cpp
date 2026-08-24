#pragma once

#include "ggml/Context.hpp"
#include "ggml/Scheduler.hpp"
#include "ggml/Tensor.hpp"
#include <functional>
#include <vector>
#include <random>
#include <cstring>
#include <stdexcept>

class MetaDevice;

class Runtime {
public:
    template<typename T>
    using Provider = std::function<std::vector<T>(std::mt19937&)>;

    struct TensorInputHash {
        std::size_t operator()(const Tensor& t) const noexcept {
            return std::hash<ggml_tensor*>()(*t);
        }
    };

    struct TensorInputEqual {
        bool operator()(const Tensor& a, const Tensor& b) const noexcept {
            return *a == *b;
        }
    };

    typedef std::unordered_map<Tensor, Provider<std::byte>, TensorInputHash, TensorInputEqual> Inputs;

    explicit Runtime(Scheduler& scheduler, Context& context, uint64_t seed = std::random_device{}(), MetaDevice* meta_device = nullptr)
        : scheduler_(scheduler), context_(context), rng_(seed), meta_device_(meta_device) {}

    // A scoped runtime shares the scheduler, context and RNG of the parent, but
    // only owns the weight buffers it allocates itself.
    Runtime(Runtime& other)
        : scheduler_(other.scheduler_), context_(other.context_), rng_(other.rng_), inputs_(other.inputs_), meta_device_(other.meta_device_)
    {
    }

    ~Runtime() {
        for (auto buffer : weight_buffers_)
            ggml_backend_buffer_free(buffer);
    }

    template <class T>
    Tensor create(const Tensor::Shape& shape, const Provider<T>& provider) {
        auto tensor = Tensor::empty<T>(*context(), shape);
        bind(tensor, provider, true);
        return tensor;
    }

    /** @brief Creates a weight tensor in a dedicated buffer of the given type and
     *  fills it once with data from the provider.
     *
     * Unlike an input (see create), a weight is not marked as a graph input: it is
     * allocated into a buffer (usage WEIGHTS) before the graph is built, so the
     * scheduler leaves it in place instead of copying it. `prepare` is invoked with
     * the tensor right before it is allocated, for backends that need per-tensor
     * state registered at allocation time (e.g. a meta split state).
     */
    template <class T>
    Tensor create_weight(
        const Tensor::Shape& shape,
        const Provider<T>& provider,
        ggml_backend_buffer_type_t buft,
        std::function<void(ggml_tensor*)> prepare = {})
    {
        auto tensor = Tensor::empty<T>(*context(), shape);

        return create_weight(
            std::move(tensor),
            [provider](std::mt19937& rng) { return to_bytes(std::move(provider(rng))); },
            buft,
            std::move(prepare));
    }

    /** @brief Allocates an existing tensor into a dedicated weight buffer of the
     *  given type and fills it once with data from the provider (see create_weight
     *  for the allocation semantics and the prepare hook). */
    Tensor create_weight(
        Tensor tensor,
        const Provider<std::byte>& provider,
        ggml_backend_buffer_type_t buft,
        std::function<void(ggml_tensor*)> prepare = {})
    {
        auto buffer = ggml_backend_buft_alloc_buffer(buft, ggml_backend_buft_get_alloc_size(buft, *tensor));
        if (buffer == nullptr)
            throw std::runtime_error("create_weight(): failed to allocate buffer");

        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        if (prepare)
            prepare(*tensor);

        if (ggml_backend_tensor_alloc(buffer, *tensor, ggml_backend_buffer_get_base(buffer)) != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buffer);
            throw std::runtime_error("create_weight(): failed to allocate tensor");
        }

        auto bytes = provider(rng_);
        if (bytes.size() != ggml_nbytes(*tensor))
            throw std::invalid_argument("create_weight(): data size mismatch: expected " + std::to_string(ggml_nbytes(*tensor)) + " bytes, but got " + std::to_string(bytes.size()));

        ggml_backend_tensor_set(*tensor, bytes.data(), 0, bytes.size());

        weight_buffers_.push_back(buffer);
        return tensor;
    }

    template <class T>
    void bind(Tensor tensor, const Provider<T>& provider, bool once = false) {
        (*tensor)->flags |= GGML_TENSOR_FLAG_INPUT;

        inputs_[tensor] = [this, tensor, provider, once](std::mt19937& rng) {
            auto values = provider(rng);

            if (once)
                this->unbind(tensor);

            return to_bytes(std::move(values));
        };
    }

    void unbind(Tensor tensor) {
        (*tensor)->flags &= ~GGML_TENSOR_FLAG_INPUT;
        inputs_.erase(tensor);
    }

    void clear() {
        inputs_.clear();
    }

    template<class T>
    std::vector<T> value(const Tensor& tensor)
    {
        constexpr auto expected = Tensor::DType<T>::value;

        if (tensor.dtype() != expected)
            throw std::invalid_argument("value(): dtype mismatch: expected " + std::string(ggml_type_name(expected)) + ", but got " + std::string(ggml_type_name(tensor.dtype())));

        std::vector<T> data(
            ggml_nelements(*tensor)
        );

        if (data.size() * sizeof(T) != ggml_nbytes(*tensor))
            throw std::invalid_argument("value(): data size mismatch: expected " + std::to_string(data.size() * sizeof(T)) + ", but got " + std::to_string(ggml_nbytes(*tensor)));

        ggml_backend_tensor_get(
            *tensor,
            data.data(),
            0,
            ggml_nbytes(*tensor)
        );

        return std::move(data);
    }

    Scheduler& scheduler() {
        return scheduler_;
    }

    Context& context() {
        return context_;
    }

    std::mt19937& rng() {
        return rng_;
    }

    /** @brief The meta (tensor-parallel) device this runtime shards weights onto, if any.
     *  Non-owning: the device must outlive the runtime. */
    MetaDevice* meta_device() const {
        return meta_device_;
    }


    Runtime(const Runtime&) = delete;
    Runtime& operator =(const Runtime&) = delete;

private:
    Scheduler& scheduler_;
    Context& context_;
    std::mt19937 rng_;
    Inputs inputs_;
    MetaDevice* meta_device_;
    std::vector<ggml_backend_buffer_t> weight_buffers_;


    template <class T>
    static std::vector<std::byte> to_bytes(std::vector<T>&& values) {
        if constexpr (std::is_same_v<T, std::byte>)
            return std::move(values);

        // Convert T[] to std::byte[]
        std::vector<std::byte> bytes(values.size() * sizeof(T));
        std::memcpy(bytes.data(), values.data(), bytes.size());
        return bytes;
    }
    friend class Graph;
};
