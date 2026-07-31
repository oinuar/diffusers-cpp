#pragma once

#include "ggml/Tensor.hpp"
#include "ggml/Runtime.hpp"

template <class AttnBackend>
class ScaledDotProductAttention {
public:
    Tensor operator()(
        Runtime& runtime,
        Tensor query,
        Tensor key,
        Tensor value,
        std::optional<Tensor> attention_mask = std::nullopt,
        std::optional<float> scale = std::nullopt
    ) {
        // Diffusers attention processor contract:
        // [B, S, H, D]

        query = query.transpose(1, 2);
        key   = key.transpose(1, 2);
        value = value.transpose(1, 2);

        AttnBackend backend;

        auto hidden_states = backend(
            runtime,
            query,
            key,
            value,
            attention_mask,
            scale
        );

        // Back to Diffusers layout:
        // [B, H, S, D] -> [B, S, H, D]

        hidden_states = hidden_states.transpose(1, 2);

        return hidden_states;
    }
};
