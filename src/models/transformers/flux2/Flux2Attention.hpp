#pragma once

#include "modules/Module.hpp"
#include "modules/Linear.hpp"
#include "modules/Dropout.hpp"
#include "models/normalization/LayerNorm.hpp"
#include "models/transformers/flux2/Flux2ParallelSelfAttention.hpp"

class Flux2Attention : public Module {
private:
    Flux2Attention(
        int64_t query_dim,
        int64_t inner_dim,
        int64_t out_dim,
        int64_t dim_head,
        float dropout,
        float eps,
        std::optional<int> added_kv_proj_dim,
        bool elementwise_affine,
        bool bias,
        bool out_bias,
        bool added_proj_bias
    )
    {
        modules["to_q"] = std::make_shared<Linear>(query_dim, inner_dim, bias);
        modules["to_k"] = std::make_shared<Linear>(query_dim, inner_dim, bias);
        modules["to_v"] = std::make_shared<Linear>(query_dim, inner_dim, bias);

        // QK Norm
        modules["norm_q"] = std::make_shared<RMSNorm>(dim_head, eps, elementwise_affine);
        modules["norm_k"] = std::make_shared<RMSNorm>(dim_head, eps, elementwise_affine);

        // Output projection
        modules["to_out.0"] = std::make_shared<Linear>(inner_dim, out_dim, out_bias);
        modules["to_out.1"] = std::make_shared<Dropout>(dropout);

        if (added_kv_proj_dim) {
            modules["norm_added_q"] = std::make_shared<RMSNorm>(dim_head, eps);
            modules["norm_added_k"] = std::make_shared<RMSNorm>(dim_head, eps);
            modules["add_q_proj"] = std::make_shared<Linear>(added_kv_proj_dim.value(), inner_dim, added_proj_bias);
            modules["add_k_proj"] = std::make_shared<Linear>(added_kv_proj_dim.value(), inner_dim, added_proj_bias);
            modules["add_v_proj"] = std::make_shared<Linear>(added_kv_proj_dim.value(), inner_dim, added_proj_bias);
            modules["to_add_out"] = std::make_shared<Linear>(inner_dim, query_dim, out_bias);
        }
    }

public:
    Flux2Attention(
        int64_t query_dim,
        int64_t heads = 8,
        int64_t dim_head = 64,
        float dropout = 0.0,
        bool bias = false,
        std::optional<int64_t> added_kv_proj_dim = std::nullopt,
        bool added_proj_bias = true,
        bool out_bias = true,
        float eps = 1e-5,
        std::optional<int64_t> out_dim = std::nullopt,
        bool elementwise_affine = true
    ) : Flux2Attention(
        query_dim,
        out_dim.value_or(dim_head * heads),
        out_dim.value_or(query_dim),
        dim_head,
        dropout,
        eps,
        added_kv_proj_dim,
        elementwise_affine,
        bias,
        out_bias,
        added_proj_bias
    )
    {
    }

    virtual std::tuple<Tensor, std::optional<Tensor>> forward(
        ggml_context* ctx,
        Tensor hidden_states,
        std::optional<Tensor> encoder_hidden_states,
        std::optional<Tensor> attention_mask = std::nullopt,
        std::optional<std::tuple<Tensor, Tensor>> image_rotary_emb = std::nullopt
    ) = 0;
};
