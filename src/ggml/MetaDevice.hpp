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

    // The split-state callback table: what the (real)
    // ggml_backend_meta_get_split_state_t callback would return, keyed by
    // the tensor pointer. It is GLOBAL across contexts -- and across every
    // allocator: the ShardedAllocator commits its plan by replacing the
    // entries of the tensors it traced (erase, then insert the new states).
    Splits& splits() { return splits_; }

    // The EFFECTIVE split state of a statically allocated tensor: its
    // planned state if the table has one, otherwise the canonical
    // MIRRORED (nr[0] = 1, n_segments = 1) -- the same default the real
    // callback returns. This is what the allocator queries when it sizes
    // a per-device slice.
    ggml_backend_meta_split_state split(const ggml_tensor* tensor) const {
        const auto it = splits_.find(tensor);

        if (it != splits_.end())
            return it->second;

        ggml_backend_meta_split_state st;
        std::memset(&st, 0, sizeof(st));
        st.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
        st.nr[0] = 1;
        st.n_segments = 1;
        return st;
    }

    MetaDevice(const MetaDevice&) = delete;
    MetaDevice& operator=(const MetaDevice&) = delete;
    MetaDevice(MetaDevice&&) = delete;
    MetaDevice& operator=(MetaDevice&&) = delete;

private:
    Splits splits_;
    size_t n_devices_;

    static ggml_backend_meta_split_state get_split_state(const ggml_tensor* tensor, void* ud) {
        return reinterpret_cast<MetaDevice*>(ud)->split(tensor);
    }
};
