#pragma once

#include "nn/Module.hpp"
#include "nn/Linear.hpp"
#include "nn/RMSNorm.hpp"
#include "models/diffusers/transformers/flux2/Flux2SwiGLU.hpp"

template <class AttnOp>
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
    ) {
        // Parallel in (QKV + MLP in) projection
        auto to_qkv_mlp_proj = std::static_pointer_cast<Linear>(modules["to_qkv_mlp_proj"]);

        hidden_states = to_qkv_mlp_proj->forward(ctx, hidden_states);

        auto parts = hidden_states.split_with_sizes({3 * inner_dim_, mlp_hidden_dim_ * mlp_mult_factor_}, -1);
        auto qkv = parts.at(0);
        auto mlp_hidden_states = parts.at(1);

        // Handle the attention logic
        auto chunks = qkv.chunk(3, -1);
        auto query = chunks.at(0);
        auto key = chunks.at(1);
        auto value = chunks.at(2);

        query = query.unflatten(-1, {heads_, -1});
        key = key.unflatten(-1, {heads_, -1});
        value = value.unflatten(-1, {heads_, -1});

        auto norm_q = std::static_pointer_cast<RMSNorm>(modules["norm_q"]);
        auto norm_k = std::static_pointer_cast<RMSNorm>(modules["norm_k"]);

        query = norm_q->forward(ctx, query);
        key = norm_k->forward(ctx, key);

        if (image_rotary_emb) {
            auto [cos, sin] = image_rotary_emb.value();

            query = apply_rotary_emb(ctx, query, cos, sin, 1);
            key = apply_rotary_emb(ctx, key, cos, sin, 1);
        }

        AttnOp dispatch_attention_fn;

        /*hidden_states = dispatch_attention_fn(
            ctx, 
            *query,
            *key,
            *value,
            // ???
            attention_mask
            //backend=self._attention_backend,
            //parallel_config=self._parallel_config,
        );*/
        hidden_states = hidden_states.flatten(2, 3);
        hidden_states = hidden_states.to(query.dtype());

        auto mlp_act_fn = std::static_pointer_cast<Flux2SwiGLU>(modules["mlp_act_fn"]);

        // Handle the feedforward (FF) logic
        mlp_hidden_states = mlp_act_fn->forward(ctx, mlp_hidden_states);

        auto to_out = std::static_pointer_cast<Linear>(modules["to_out"]);

        // Concatenate and parallel output projection
        hidden_states = Tensor::cat({hidden_states, mlp_hidden_states}, -1);
        hidden_states = to_out->forward(ctx, hidden_states);

        return hidden_states;
    }

protected:
    int64_t inner_dim_, mlp_hidden_dim_;
};
