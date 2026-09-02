#pragma once

#include "nn/Module.hpp"
#include "nn/Linear.hpp"
#include "nn/modules/normalization/GroupNorm.hpp"

template <class AttnOp>
class Attention : public Module {
public:
    Attention(
        int64_t query_dim,
        int64_t heads,
        int64_t dim_head = -1,
        float dropout = 0.0f,
        bool bias = false,
        bool residual_connection = false,
        std::optional<int64_t> norm_num_groups = std::nullopt,
        float eps = 1e-6,
        float rescale_output_factor = false,
        bool upcast_softmax = false
    );

    Tensor forward(Scope scope, Tensor hidden_states);

private:
    int64_t heads_;
    int64_t dim_head_;
    int64_t inner_dim_;
    bool residual_connection_;
    bool upcast_softmax_;
    std::optional<int64_t> norm_num_groups_;
    float rescale_output_factor_;
};

#include "diffusers/models/attention_processor/Attention.inl"
