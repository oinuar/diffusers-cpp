#pragma once

#include "models/transformers/flux2/Flux2ParallelSelfAttention.hpp"
#include "models/embeddings/funcs.hpp"

template <class AttnOp>
class Flux2ParallelSelfAttnProcessor : public Flux2ParallelSelfAttention {
public:
    Flux2ParallelSelfAttnProcessor(
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
        query_dim,
        heads,
        dim_head,
        dropout,
        bias,
        out_bias,
        eps,
        out_dim,
        elementwise_affine,
        mlp_ratio,
        mlp_mult_factor
    ), heads_(heads), mlp_mult_factor_(mlp_mult_factor)
    {
    }

    Tensor forward(
        ggml_context* ctx,
        Tensor hidden_states,
        std::optional<Tensor> attention_mask = std::nullopt,
        std::optional<std::tuple<Tensor, Tensor>> image_rotary_emb = std::nullopt
    )
    {
        // Parallel in (QKV + MLP in) projection
        auto to_qkv_mlp_proj = std::static_pointer_cast<Linear>(modules["to_qkv_mlp_proj"]);

        hidden_states = to_qkv_mlp_proj->forward(ctx, hidden_states);

        // TODO: no split
        auto parts = std::vector<Tensor>({hidden_states, hidden_states}); // hidden_states.split({3 * inner_dim_, mlp_hidden_dim_ * mlp_mult_factor_}, -1);
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
        hidden_states = hidden_states.flatten({2, 3});
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

private:
    int64_t heads_, mlp_mult_factor_;
};
