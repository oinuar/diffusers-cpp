#pragma once

#include <cmath>
#include <vector>

#include "nn/Module.hpp"
#include "ggml/Context.hpp"

class Flux2PosEmbed : public Module {
public:
    Flux2PosEmbed(
        int64_t theta,
        const std::vector<int64_t>& axes_dim)
        : theta_(theta), axes_dim_(axes_dim)
    {
    }

    Tensor forward(
        Context& context,
        Tensor x,
        Tensor position_ids)
    {
        // GGML RoPE expects positions IDs to be 32b integers.
        if (position_ids.dtype() != GGML_TYPE_I32)
            position_ids = position_ids.to(GGML_TYPE_I32);

        std::vector<Tensor> rotated_sections;

        int64_t offset = 0;

        for (int axis = 0; axis < axes_dim_.size(); ++axis) {
            const int64_t axis_dim = axes_dim_[axis];

            //
            // Extract the part of the head dimension belonging to this axis.
            //
            // For example with axes_dim={2,2,2,2}:
            //
            //   axis 0 -> x[..., 0:2]  (T)
            //   axis 1 -> x[..., 2:4]  (H)
            //   axis 2 -> x[..., 4:6]  (W)
            //   axis 3 -> x[..., 6:8]  (L)
            //
            auto x_axis = x[{Tensor::Slice::ellipsis(), Tensor::Slice::range(offset, offset + axis_dim)}];

            //
            // Extract the corresponding position IDs.
            //
            // position_ids layout:
            //
            //   [
            //     [T0, H0, W0, L0],
            //     [T1, H1, W1, L1],
            //     ...
            //   ]
            //
            // Each RoPE operation receives one independent axis:
            //
            //   T positions -> T dimensions
            //   H positions -> H dimensions
            //   W positions -> W dimensions
            //   L positions -> L dimensions
            //
            auto pos_axis = position_ids[{Tensor::Slice::all(), Tensor::Slice::index(axis)}];

            // GGML RoPE expects position IDs to be a 1D tensor (vector).
            if (!ggml_is_vector(*pos_axis))
                pos_axis = pos_axis.flatten();

            //
            // Apply standard interleaved RoPE.
            //
            // Flux2 uses:
            //
            //   x.reshape(..., D/2, 2)
            //
            // which corresponds to GGML_ROPE_TYPE_NORMAL:
            //
            //   (x0,x1), (x2,x3), ...
            //
            auto rope = ggml_rope_ext(
                *context,
                *x_axis,
                *pos_axis,
                nullptr,                // no YaRN frequency tensor

                axis_dim,

                GGML_ROPE_TYPE_NORMAL,

                0,                      // n_ctx_orig
                theta_,
                1.0f,                   // freq_scale
                0.0f,                   // ext_factor
                1.0f,                   // attn_factor
                0.0f,
                0.0f
            );

            rotated_sections.emplace_back(Tensor(*context, rope, x_axis.shape()));
            offset += axis_dim;
        }

        //
        // Reassemble:
        //
        //   [rope(T) | rope(H) | rope(W) | rope(L)]
        //
        // restoring the original head dimension.
        //
        return Tensor::cat(rotated_sections, -1);
    }

private:
    int64_t theta_;
    std::vector<int64_t> axes_dim_;
};
