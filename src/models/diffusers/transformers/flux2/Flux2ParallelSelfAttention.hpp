#pragma once

#include "nn/Module.hpp"
#include "nn/Linear.hpp"
#include "models/diffusers/transformers/flux2/Flux2SwiGLU.hpp"
#include "models/normalization/RMSNorm.hpp"

class Flux2ParallelSelfAttention : public Module {
private:
    Flux2ParallelSelfAttention(
        int64_t dim_head,
        int64_t out_dim,
        int64_t query_dim,
        int64_t inner_dim,
        int64_t mlp_hidden_dim,
        int64_t mlp_mult_factor,
        float eps,
        bool elementwise_affine,
        bool bias,
        bool out_bias
    ) : inner_dim_(inner_dim), mlp_hidden_dim_(mlp_hidden_dim) {
        // Fused QKV projections + MLP input projection
        modules["to_qkv_mlp_proj"] = std::make_shared<Linear>(query_dim, inner_dim * 3 + mlp_hidden_dim * mlp_mult_factor, bias);
        modules["mlp_act_fn"] = std::make_shared<Flux2SwiGLU>();
        
        // QK Norm
        modules["norm_q"] = std::make_shared<RMSNorm>(dim_head, eps, elementwise_affine);
        modules["norm_k"] = std::make_shared<RMSNorm>(dim_head, eps, elementwise_affine);

        // Fused attention output projection + MLP output projection
        modules["to_out"] = std::make_shared<Linear>(inner_dim + mlp_hidden_dim, out_dim, out_bias);
    }

public:
    Flux2ParallelSelfAttention(
        int64_t query_dim,
        int64_t heads = 8,
        int64_t dim_head = 64,
        float dropout = 0.0,
        bool bias = false,
        bool out_bias = true,
        float eps = 1e-5,
        std::optional<int64_t> out_dim = std::nullopt,
        bool elementwise_affine = true,
        float mlp_ratio = 4.0,
        int64_t mlp_mult_factor = 2
    ) : Flux2ParallelSelfAttention(
            dim_head,
            out_dim.value_or(query_dim),
            query_dim,
            out_dim.value_or(dim_head * heads),
            query_dim * mlp_ratio,
            mlp_mult_factor,
            eps,
            elementwise_affine,
            bias,
            out_bias
        )
    {
    }

    virtual Tensor forward(
        ggml_context* ctx,
        Tensor hidden_states,
        std::optional<Tensor> attention_mask = std::nullopt,
        std::optional<std::tuple<Tensor, Tensor>> image_rotary_emb = std::nullopt
    ) = 0;

protected:
    int64_t inner_dim_, mlp_hidden_dim_;
};
