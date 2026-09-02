#pragma once

#include "nn/Module.hpp"
#include "nn/Identity.hpp"
#include "nn/SiLU.hpp"
#include <optional>

template <class ActFn = SiLU, class PostActFn = Identity>
class TimestepEmbedding : public Module {
public:
    TimestepEmbedding(
        int64_t in_channels,
        int64_t time_embed_dim,
        std::optional<int64_t> out_dim = {},
        std::optional<int64_t> cond_proj_dim = {},
        bool sample_proj_bias = true
    );
    
    Tensor forward(Scope scope, Tensor sample, std::optional<Tensor> condition = std::nullopt);
};

#include "diffusers/models/embeddings/TimestepEmbedding.inl"
