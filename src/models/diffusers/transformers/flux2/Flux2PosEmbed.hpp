#pragma once

#include <cmath>
#include <vector>

#include "nn/Module.hpp"

class Flux2PosEmbed : public Module {
public:
    Flux2PosEmbed(int64_t theta, const std::vector<int64_t>& axes_dim)
        : theta_(theta), axes_dim_(axes_dim)
    {
    }

    std::pair<Tensor, Tensor> forward(ggml_context* ctx, Tensor ids) {
        std::vector<Tensor> cos_out;
        std::vector<Tensor> sin_out;

        auto pos = ids.to(GGML_TYPE_F32);

        for (size_t i = 0; i < axes_dim_.size(); ++i) {
            auto [cos, sin] = get_1d_rotary_pos_embed(
                ctx,
                axes_dim_[i],
                pos[{Tensor::Slice::ellipsis(), Tensor::Slice::index(i)}],
                theta_,
                true,
                true
            );

            cos_out.push_back(cos);
            sin_out.push_back(sin);
        }

        return {
            Tensor::cat(cos_out, -1),
            Tensor::cat(sin_out, -1)
        };
    }

private:
    int64_t theta_;
    std::vector<int64_t> axes_dim_;

    static Tensor repeat_interleave_last_dim(Tensor x) {
        std::vector<Tensor> cols;
        cols.reserve(x.shape()[1] * 2);

        for (auto i = 0; i < x.shape()[1]; ++i) {
            auto col = x[{Tensor::Slice::all(), Tensor::Slice::range(i, i + 1)}];
            cols.push_back(col);
            cols.push_back(col);
        }

        return Tensor::cat(cols, -1);
    }

    static std::pair<Tensor, Tensor> get_1d_rotary_pos_embed(
        ggml_context* ctx,
        int64_t dim,
        Tensor pos,
        float theta = 10000.f,
        bool repeat_interleave_real = true,
        bool use_real = true)
    {
        const int64_t half_dim = dim / 2;

        auto freqs = Tensor::arange(ctx, 0, half_dim);
        freqs = freqs / static_cast<float>(half_dim);

        freqs = pow(theta, -freqs);

        auto args = pos.unsqueeze(-1) * freqs.unsqueeze(0);

        auto cos_val = cos(args);
        auto sin_val = sin(args);

        std::cerr << "args      " << args.shape().to_string() << '\n';
        std::cerr << "cos       " << cos_val.shape().to_string() << '\n';

        if (repeat_interleave_real) {
            cos_val = repeat_interleave_last_dim(cos_val);
            sin_val = repeat_interleave_last_dim(sin_val);

            std::cerr << "repeat    " << cos_val.shape().to_string() << '\n';
        }

        return {cos_val, sin_val};
    }
};
