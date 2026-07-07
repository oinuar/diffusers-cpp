#pragma once

#include <optional>

#include "nn/Module.hpp"
#include "nn/Linear.hpp"
#include "nn/Identity.hpp"
#include "nn/SiLU.hpp"

template <class ActFn = Identity, class PostActFn = Identity>
class TimestepEmbedding : public Module {
public:
    TimestepEmbedding(
        int64_t in_channels,
        int64_t time_embed_dim,
        bool sample_proj_bias = true,
        std::optional<int64_t> out_dim = {},
        std::optional<const char*> post_act_fn = {},
        std::optional<int64_t> cond_proj_dim = {}
    ) {
        modules["linear_1"] = std::make_shared<Linear>(in_channels, time_embed_dim, sample_proj_bias);

        if (cond_proj_dim)
            modules["cond_proj"] = std::make_shared<Linear>(cond_proj_dim.value(), in_channels, false);

        modules["act"] = std::make_shared<ActFn>();

        modules["linear_2"] = std::make_shared<Linear>(time_embed_dim, out_dim.value_or(time_embed_dim), sample_proj_bias);

        modules["post_act"] = std::make_shared<PostActFn>();
    }
    
    Tensor forward(ggml_context* ctx, Tensor sample, std::optional<Tensor> condition = std::nullopt) {
        if (condition) {
            auto cond_proj = std::static_pointer_cast<Linear>(modules["cond_proj"]);

            sample = sample + cond_proj->forward(ctx, condition.value());
        }

        auto linear_1 = std::static_pointer_cast<Linear>(modules["linear_1"]);
        auto act = std::static_pointer_cast<ActFn>(modules["act"]);
        auto linear_2 = std::static_pointer_cast<Linear>(modules["linear_2"]);
        auto post_act = std::static_pointer_cast<PostActFn>(modules["post_act"]);

        sample = linear_1->forward(ctx, sample);
        sample = act->forward(ctx, sample);
        sample = linear_2->forward(ctx, sample);
        sample = post_act->forward(ctx, sample);

        return sample;
    }
};
