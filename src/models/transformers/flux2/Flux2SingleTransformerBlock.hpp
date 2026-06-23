#pragma once

#include "modules/Module.hpp"
#include "modules/Linear.hpp"
#include "models/normalization/LayerNorm.hpp"
#include "models/transformers/flux2/Flux2ParallelSelfAttnProcessor.hpp"
#include "models/transformers/flux2/Flux2Modulation.hpp"
#include "models/attention/SoftmaxAttnOp.hpp"

class Flux2SingleTransformerBlock : public Module {
public:
    Flux2SingleTransformerBlock(
        int64_t dim,
        int64_t num_attention_heads,
        int64_t attention_head_dim,
        float mlp_ratio = 3.0,
        float eps = 1e-6,
        bool bias = false
    ) {
        modules["norm"] = std::make_shared<LayerNorm>(dim, eps, false);
        modules["attn"] = std::make_shared<Flux2ParallelSelfAttnProcessor<SoftmaxAttnOp>>(
            dim,
            num_attention_heads,
            attention_head_dim,
            0.0,
            bias,
            bias,
            eps,
            dim,
            true,
            mlp_ratio,
            2
        );
    }

    std::tuple<Tensor, Tensor> forward(
        ggml_context* ctx,
        Tensor hidden_states,
        std::optional<Tensor> encoder_hidden_states,
        Tensor temb_mob,
        std::optional<std::tuple<Tensor, Tensor>> image_rotary_emb = std::nullopt,
        bool split_hidden_states = false,
        std::optional<int64_t> text_seq_len = std::nullopt
    ) {
        auto norm = std::static_pointer_cast<LayerNorm>(modules["norm"]);
        auto attn = std::static_pointer_cast<Flux2ParallelSelfAttention>(modules["attn"]);

        if (encoder_hidden_states) {
            text_seq_len = encoder_hidden_states.value().shape()[1];
            hidden_states = Tensor::cat({encoder_hidden_states.value(), hidden_states}, 1);
        }

        auto [mod_shift, mod_scale, mod_gate] = Flux2Modulation::split(temb_mob, 1)[0];

        auto norm_hidden_states = norm->forward(ctx, hidden_states);
        norm_hidden_states = (1.0f + mod_scale) * norm_hidden_states + mod_shift;

        auto attn_output = attn->forward(ctx, norm_hidden_states, std::nullopt, image_rotary_emb);
        hidden_states = hidden_states + mod_gate * attn_output;

        if (hidden_states.dtype() == GGML_TYPE_F16)
            hidden_states = hidden_states.clip(-65504, 65504);

        if (split_hidden_states) {
            if (text_seq_len) {
                encoder_hidden_states = hidden_states.narrow(1, 0, text_seq_len.value()); // Python: hidden_states[:, :text_seq_len]
                hidden_states = hidden_states.narrow(1, text_seq_len.value(), hidden_states.shape()[1] - text_seq_len.value()); // Python: hidden_states[:, text_seq_len:]
                return {encoder_hidden_states.value(), hidden_states};
            }

            return {hidden_states, hidden_states};
        }

        return {hidden_states, Tensor()};
    }
};
