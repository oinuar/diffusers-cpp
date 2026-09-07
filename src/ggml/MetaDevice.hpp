#pragma once

#include "ggml/Tensor.hpp"
#include "ggml/Device.hpp"
#include "nn/Parameter.hpp"
#include <unordered_map>
#include <vector>
#include <map>
#include <stdexcept>
#include <cstring>
#include <iostream>

/** @brief A virtual device that shards tensors across N underlying devices
 *  (e.g. multiple GPUs).
 *
 * Wraps ggml's meta backend. Statically-allocated tensors (weights) are given an
 * explicit SplitState via split() before they are allocated into a meta buffer;
 * compute tensors are split automatically to stay compatible with their operands.
 *
 * @note ggml caches the meta device (and its split-state callback) for the process
 *  lifetime and keeps a pointer to this object's registry. Keep the MetaDevice
 *  alive for as long as the meta device / backend is in use.
 */
class MetaDevice : public Device {
public:
    typedef std::map<const ggml_tensor*, ggml_backend_meta_split_state> Splits;

    explicit MetaDevice(std::vector<ggml_backend_dev_t> devices)
        : Device(ggml_backend_meta_device(devices.data(), devices.size(), get_split_state, this)),
          n_devices_(devices.size())
    {
    }

    static MetaDevice all(enum ggml_backend_dev_type type) {
        std::vector<ggml_backend_dev_t> devices;

        for (auto i = 0; i < ggml_backend_dev_count(); ++i) {
            auto dev = ggml_backend_dev_get(i);

            if (ggml_backend_dev_type(dev) == type)
                devices.push_back(dev);
        }

        return MetaDevice(devices);
    }

    size_t count() const {
        return n_devices_;
    }

    Splits& splits() {
        return splits_;
    }

    MetaDevice(const MetaDevice&) = delete;
    MetaDevice& operator=(const MetaDevice&) = delete;
    MetaDevice(MetaDevice&&) = delete;
    MetaDevice& operator=(MetaDevice&&) = delete;

private:
    Splits splits_;
    size_t n_devices_;

    static ggml_backend_meta_split_state get_split_state(const ggml_tensor* tensor, void* ud) {
        auto self = reinterpret_cast<MetaDevice*>(ud);
        auto it = self->splits().find(tensor);

        if (it == std::end(self->splits())) {
            ggml_backend_meta_split_state st;
            std::memset(&st, 0, sizeof(st));

            st.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
            st.nr[0] = 1;
            st.n_segments = 1;

            return st;
        }

        return it->second;
    }
};
