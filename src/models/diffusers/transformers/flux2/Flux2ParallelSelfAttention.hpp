#pragma once

#include "nn/Module.hpp"
#include "nn/Linear.hpp"
#include "nn/RMSNorm.hpp"
#include "models/diffusers/transformers/flux2/Flux2SwiGLU.hpp"

template <class AttnOp>
class Flux2ParallelSelfAttention : public Module {
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
        int64_t mlp_mult_factor = 2)
    {
        head_dim_ = dim_head;
        inner_dim_ = out_dim.value_or(dim_head * heads);
        query_dim_ = query_dim;
        out_dim_ = out_dim.value_or(query_dim);
        heads_ = out_dim ? *out_dim / dim_head : heads;

        use_bias_ = bias;
        dropout_ = dropout;

        mlp_ratio_ = mlp_ratio;
        mlp_hidden_dim_ = int(query_dim * mlp_ratio_);
        mlp_mult_factor_ = mlp_mult_factor;

        // Fused QKV projections + MLP input projection
        modules["to_qkv_mlp_proj"] = std::make_shared<Linear>(query_dim_, inner_dim_ * 3 + mlp_hidden_dim_ * mlp_mult_factor, bias);
        modules["mlp_act_fn"] = std::make_shared<Flux2SwiGLU>();
        
        // QK Norm
        modules["norm_q"] = std::make_shared<RMSNorm>(dim_head, eps, elementwise_affine);
        modules["norm_k"] = std::make_shared<RMSNorm>(dim_head, eps, elementwise_affine);

        // Fused attention output projection + MLP output projection
        modules["to_out"] = std::make_shared<Linear>(inner_dim_ + mlp_hidden_dim_, out_dim_, out_bias);
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
    int64_t head_dim_;
    int64_t inner_dim_;
    int64_t query_dim_;
    int64_t out_dim_;
    int64_t heads_;

    bool use_bias_;
    float dropout_;

    float mlp_ratio_;
    int64_t mlp_hidden_dim_;
    int64_t mlp_mult_factor_;

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
