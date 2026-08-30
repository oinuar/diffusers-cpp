#pragma once

#include "nn/Module.hpp"
#include "nn/modules/normalization/LayerNorm.hpp"

template <class NormFn = LayerNorm>
class AdaLayerNormContinuous : public Module {
public:
    AdaLayerNormContinuous(
        int64_t embedding_dim,
        int64_t conditioning_embedding_dim,
        bool elementwise_affine = true,
        float eps = 1e-5f,
        bool bias = true
    );

    Tensor forward(Context& context, Tensor hidden_states, Tensor conditioning_embedding);
};

#include "diffusers/models/normalization/AdaLayerNormContinuous.inl"
