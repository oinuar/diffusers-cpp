#pragma once

#include "modules/Module.hpp"
#include "modules/Linear.hpp"
#include "models/transformers/flux2/Flux2SwiGLU.hpp"

class Flux2FeedForward : public Module {
private:
    Flux2FeedForward(
        int64_t dim,
        int64_t dim_out,
        float mult,
        int64_t inner_dim,
        bool bias
    ) {
        modules["linear_in"] = std::make_shared<Linear>(dim, inner_dim * 2, bias);
        modules["act_fn"] = std::make_shared<Flux2SwiGLU>();
        modules["linear_out"] = std::make_shared<Linear>(inner_dim, dim_out, bias);
    }

public:
    Flux2FeedForward(
        int64_t dim,
        std::optional<int64_t> dim_out = std::nullopt,
        float mult = 3.0,
        std::optional<int64_t> inner_dim = std::nullopt,
        bool bias = false
    ) :
        Flux2FeedForward(
            dim,
            dim_out.value_or(dim),
            mult,
            inner_dim.value_or(dim * mult),
            bias
        )
    {
    }

    Tensor forward(ggml_context* ctx, Tensor x) {
        auto linear_in = std::static_pointer_cast<Linear>(modules["linear_in"]);
        auto act_fn = std::static_pointer_cast<Flux2SwiGLU>(modules["act_fn"]);
        auto linear_out = std::static_pointer_cast<Linear>(modules["linear_out"]);

        x = linear_in->forward(ctx, x);
        x = act_fn->forward(ctx, x);
        x = linear_out->forward(ctx, x);
        return x;
    }
};
