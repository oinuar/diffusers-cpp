#pragma once

#include "nn/Module.hpp"

template <class AttnOp>
class Attention : public Module {
public:
    Attention(
        int64_t query_dim,
        int64_t heads = 1,
        int64_t dim_head = -1,
        float dropout = 0.0f,
        bool bias = false
    );

    Tensor forward(
        ggml_context* ctx,
        Tensor hidden_states
    );

private:
    int64_t heads_;
    int64_t dim_head_;
    int64_t inner_dim_;
};

#include "diffusers/models/attention_processor/Attention.inl"