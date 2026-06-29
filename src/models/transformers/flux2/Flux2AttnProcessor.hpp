#pragma once

#include "models/transformers/flux2/Flux2Attention.hpp"
#include "models/embeddings/funcs.hpp"

template <class AttnOp>
class Flux2AttnProcessor : public Flux2Attention {
public:
    Flux2AttnProcessor(
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
        heads,
        dim_head,
        dropout,
        bias,
        added_kv_proj_dim,
        added_proj_bias,
        out_bias,
        eps,
        out_dim,
        elementwise_affine
    ), heads_(heads), added_kv_proj_dim_(added_kv_proj_dim)
    {
    }

    virtual std::tuple<Tensor, std::optional<Tensor>> forward(
        ggml_context* ctx,
        Tensor hidden_states,
        std::optional<Tensor> encoder_hidden_states,
        std::optional<Tensor> attention_mask = std::nullopt,
        std::optional<std::tuple<Tensor, Tensor>> image_rotary_emb = std::nullopt
    ) {
        // _get_qkv_projections(...) when use_fused_projections=false
        auto to_q = std::static_pointer_cast<Linear>(modules["to_q"]);
        auto to_k = std::static_pointer_cast<Linear>(modules["to_k"]);
        auto to_v = std::static_pointer_cast<Linear>(modules["to_v"]);

        auto query = to_q->forward(ctx, hidden_states);
        auto key = to_k->forward(ctx, hidden_states);
        auto value = to_v->forward(ctx, hidden_states);

        std::optional<Tensor> encoder_query, encoder_key, encoder_value;

        if (encoder_hidden_states && added_kv_proj_dim_) {
            auto add_q_proj = std::static_pointer_cast<Linear>(modules["add_q_proj"]);
            auto add_k_proj = std::static_pointer_cast<Linear>(modules["add_k_proj"]);
            auto add_v_proj = std::static_pointer_cast<Linear>(modules["add_v_proj"]);

            encoder_query = add_q_proj->forward(ctx, encoder_hidden_states.value());
            encoder_key = add_k_proj->forward(ctx, encoder_hidden_states.value());
            encoder_value = add_v_proj->forward(ctx, encoder_hidden_states.value());
        }
        // end of _get_qkv_projections(...)

        query = query.unflatten(-1, {heads_, -1});
        key = key.unflatten(-1, {heads_, -1});
        value = value.unflatten(-1, {heads_, -1});

        auto norm_q = std::static_pointer_cast<RMSNorm>(modules["norm_q"]);
        auto norm_k = std::static_pointer_cast<RMSNorm>(modules["norm_k"]);

        query = norm_q->forward(ctx, query);
        key = norm_k->forward(ctx, key);

        if (added_kv_proj_dim_) {
            encoder_query = encoder_query.value().unflatten(-1, {heads_, -1});
            encoder_key = encoder_key.value().unflatten(-1, {heads_, -1});
            encoder_value = encoder_value.value().unflatten(-1, {heads_, -1});

            auto norm_added_q = std::static_pointer_cast<RMSNorm>(modules["norm_added_q"]);
            auto norm_added_k = std::static_pointer_cast<RMSNorm>(modules["norm_added_k"]);

            encoder_query = norm_added_q->forward(ctx, encoder_query.value());
            encoder_key = norm_added_k->forward(ctx, encoder_key.value());

            query = Tensor::cat({encoder_query.value(), query}, 1);
            key = Tensor::cat({encoder_key.value(), key}, 1);
            value = Tensor::cat({encoder_value.value(), value}, 1);
        }

        if (image_rotary_emb) {
            auto [cos, sin] = image_rotary_emb.value();

            query = apply_rotary_emb(ctx, query, cos, sin, 1);
            key = apply_rotary_emb(ctx, key, cos, sin, 1);
        }

        AttnOp dispatch_attention_fn;

        /*hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            // ???
            attention_mask
            //backend=self._attention_backend,
            //parallel_config=self._parallel_config,
        );*/
        hidden_states = hidden_states.flatten(2, 3);
        hidden_states = hidden_states.to(query.dtype());

        if (encoder_hidden_states) {
            auto to_add_out = std::static_pointer_cast<Linear>(modules["to_add_out"]);

            auto parts = hidden_states.split_with_sizes({
                encoder_hidden_states.value().shape()[1],
                hidden_states.shape()[1] - encoder_hidden_states.value().shape()[1]
            }, 1);
            
            encoder_hidden_states = parts.at(0);
            hidden_states = parts.at(1);

            encoder_hidden_states = to_add_out->forward(ctx, encoder_hidden_states.value());
        }

        auto to_out0 = std::static_pointer_cast<Linear>(modules["to_out.0"]);
        auto to_out1 = std::static_pointer_cast<Dropout>(modules["to_out.1"]);

        hidden_states = to_out0->forward(ctx, hidden_states);
        hidden_states = to_out1->forward(ctx, hidden_states);

        if (encoder_hidden_states)
            return {hidden_states, encoder_hidden_states.value()};

        return {hidden_states, std::nullopt};
    }

private:
    int64_t heads_;
    std::optional<int64_t> added_kv_proj_dim_;
};
