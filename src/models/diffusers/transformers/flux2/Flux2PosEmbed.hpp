#pragma once

#include <cmath>
#include <tuple>
#include <vector>

#include "nn/Module.hpp"
#include "models/embeddings/funcs.hpp"
#include "ggml-backend.h"

class Flux2PosEmbed : public Module {
public:
    Flux2PosEmbed(int64_t theta, const std::vector<int64_t>& axes_dim)
        : theta_(theta), axes_dim_(axes_dim)
    {
    }

    std::pair<Tensor, Tensor> forward(ggml_context* ctx, Tensor ids) {
        // ids shape: [S, len(axes_dim)]  (int32 position indices)
        auto pos = ids.to(GGML_TYPE_F32);

        std::vector<Tensor> cos_out, sin_out;

        for (size_t i = 0; i < axes_dim_.size(); ++i) {
            auto [cos, sin] = get_1d_rotary_pos_embed(
                ctx,
                static_cast<int64_t>(axes_dim_[i]),
                pos[{Tensor::Slice::ellipsis(), Tensor::Slice::index(i)}], // [S]
                static_cast<double>(theta_)
            );

            cos_out.push_back(cos);
            sin_out.push_back(sin);
        }

        auto freqs_cos = Tensor::cat(cos_out, -1);
        auto freqs_sin = Tensor::cat(sin_out, -1);

        return {freqs_cos, freqs_sin};
    }

private:
    int64_t theta_;
    std::vector<int64_t> axes_dim_;
};
