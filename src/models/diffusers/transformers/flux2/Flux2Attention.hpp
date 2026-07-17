#pragma once

#include "nn/Module.hpp"
#include "nn/ModuleList.hpp"
#include "nn/Linear.hpp"
#include "nn/Dropout.hpp"
#include "nn/LayerNorm.hpp"

template <class AttnOp>
class Flux2Attention : public Module {
private:
    Flux2Attention(
        int64_t heads,
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
    ) : heads_(heads), added_kv_proj_dim_(added_kv_proj_dim)
    {
        modules["to_q"] = std::make_shared<Linear>(query_dim, inner_dim, bias);
        modules["to_k"] = std::make_shared<Linear>(query_dim, inner_dim, bias);
        modules["to_v"] = std::make_shared<Linear>(query_dim, inner_dim, bias);

        // QK Norm
        modules["norm_q"] = std::make_shared<RMSNorm>(dim_head, eps, elementwise_affine);
        modules["norm_k"] = std::make_shared<RMSNorm>(dim_head, eps, elementwise_affine);

        // Output projection
        modules["to_out"] = std::shared_ptr<ModuleList>(new ModuleList({
            std::make_shared<Linear>(inner_dim, out_dim, out_bias),
            std::make_shared<Dropout>(dropout)
        }));
        
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
        heads,
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

            query = apply_rotary_emb(ctx, query, cos, sin);
            key = apply_rotary_emb(ctx, key, cos, sin);
        }

        AttnOp dispatch_attention_fn;

        hidden_states = dispatch_attention_fn(
            ctx,
            query,
            key,
            value,
            attention_mask
        );

        hidden_states = hidden_states.contiguous();
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

        auto to_out = std::static_pointer_cast<ModuleList>(modules["to_out"]);

        hidden_states = std::static_pointer_cast<Linear>((*to_out)[0])->forward(ctx, hidden_states);
        hidden_states = std::static_pointer_cast<Dropout>((*to_out)[1])->forward(ctx, hidden_states);

        if (encoder_hidden_states)
            return {hidden_states, encoder_hidden_states.value()};

        return {hidden_states, std::nullopt};
    }

private:
    int64_t heads_;
    std::optional<int> added_kv_proj_dim_;

    static Tensor apply_rotary_emb(
        ggml_context* ctx,
        Tensor x,
        const Tensor& cos,
        const Tensor& sin
    ) {
        const auto head_dim = x.shape()[-1];

        // (..., D) -> (..., D/2, 2)
        auto t = x.unflatten(-1, {head_dim / 2, 2});

        // Extract real/imag parts of each pair.
        auto real = t[{Tensor::Slice::ellipsis(), Tensor::Slice::index(0)}];
        auto imag = t[{Tensor::Slice::ellipsis(), Tensor::Slice::index(1)}];

        // Rotate: (a, b) -> (-b, a)
        auto rotated = Tensor::cat({
            (-imag).unsqueeze(-1),
            real.unsqueeze(-1),
        }, -1);

        // (..., D/2, 2) -> (..., D)
        rotated = rotated.flatten(-2, -1);

        return x * cos + rotated * sin;
    }
};
