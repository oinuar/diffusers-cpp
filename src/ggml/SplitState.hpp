#pragma once

#include "ggml-backend.h"
#include <stdexcept>

/** @brief Describes how a statically-allocated tensor is sharded across the
 *  devices of a meta (tensor-parallel) device.
 *
 * Only weight tensors (statically allocated) are given an explicit SplitState; the
 * meta backend derives compatible split states for compute tensors automatically.
 * A tensor with no registered SplitState defaults to #mirrored (a full copy on
 * every device).
 */
struct SplitState {
    ggml_backend_meta_split_state state;

    /** @brief A full copy of the tensor on every device (the default). */
    static SplitState mirrored() {
        SplitState s;
        s.state.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
        s.state.ne[0] = 0;
        s.state.nr[0] = 1;
        s.state.n_segments = 1;
        return s;
    }

    /** @brief Splits one dimension evenly across `n_devices` devices.
     *
     * `extent` elements along ggml dimension `ggml_axis` are divided into
     * `n_devices` contiguous slices, one per device.
     *
     * @param ggml_axis ggml dimension to split (0 = fastest, < GGML_MAX_DIMS).
     * @param extent    size of the tensor along `ggml_axis`.
     * @param n_devices number of devices in the meta device.
     */
    static SplitState split(int64_t ggml_axis, int64_t extent, int n_devices) {
        if (n_devices < 1)
            throw std::invalid_argument("SplitState::split(): n_devices must be >= 1");

        if (ggml_axis < 0 || ggml_axis >= GGML_MAX_DIMS)
            throw std::invalid_argument("SplitState::split(): ggml_axis out of range");

        if (extent % n_devices != 0)
            throw std::invalid_argument("SplitState::split(): extent must be divisible by n_devices");

        SplitState s;
        s.state.axis = static_cast<ggml_backend_meta_split_axis>(ggml_axis);
        s.state.n_segments = 1;
        s.state.nr[0] = 1;
        const int64_t per_device = extent / n_devices;
        for (int i = 0; i < n_devices; ++i)
            s.state.ne[i] = per_device;

        return s;
    }
};
