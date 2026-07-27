#pragma once

#include "nn/Module.hpp"
#include "nn/modules/normalization/LayerNorm.hpp"
#include "diffusers/models/transformers/flux2/Flux2Attention.hpp"
#include "diffusers/models/transformers/flux2/Flux2FeedForward.hpp"
#include "diffusers/models/transformers/flux2/Flux2Modulation.hpp"

template <class AttnOp>
class Flux2TransformerBlock : public Module {
public:
    Flux2TransformerBlock(
        int64_t dim,
        int64_t num_attention_heads,
        int64_t attention_head_dim,
        float mlp_ratio = 3.0,
        float eps = 1e-6,
        bool bias = false
    ) {
        int64_t mlp_hidden_dim = dim * mlp_ratio;

        modules["norm1"] = std::make_shared<LayerNorm>(dim, eps, false);
        modules["norm1_context"] = std::make_shared<LayerNorm>(dim, eps, false);

        modules["attn"] = std::make_shared<Flux2Attention<AttnOp>>(
            dim,
            num_attention_heads,
            attention_head_dim,
            0.0,
            bias,
            dim,
            bias,
            bias,
            eps,
            dim
        );

        modules["norm2"] = std::make_shared<LayerNorm>(dim, eps, false);
        modules["ff"] = std::make_shared<Flux2FeedForward>(dim, dim, mlp_ratio, std::nullopt, bias);

        modules["norm2_context"] = std::make_shared<LayerNorm>(dim, eps, false);
        modules["ff_context"] = std::make_shared<Flux2FeedForward>(dim, dim, mlp_ratio, std::nullopt, bias);
    }

    std::tuple<Tensor, Tensor> forward(
        Runtime& runtime,
        Tensor hidden_states,
        Tensor encoder_hidden_states,
        Tensor temb_mod_img,
        Tensor temb_mod_txt,
        std::optional<std::pair<std::shared_ptr<Flux2PosEmbed>, Tensor>> image_rotary_emb = std::nullopt
    ) {
        // Modulation parameters shape: [1, 1, self.dim]
        auto split = Flux2Modulation::split(temb_mod_img, 2);
        auto [shift_msa, scale_msa, gate_msa] = split.at(0);
        auto [shift_mlp, scale_mlp, gate_mlp] = split.at(1);

        split = Flux2Modulation::split(temb_mod_txt, 2);
        auto [c_shift_msa, c_scale_msa, c_gate_msa] = split.at(0);
        auto [c_shift_mlp, c_scale_mlp, c_gate_mlp] = split.at(1);

        // Img stream
        auto norm1 = std::static_pointer_cast<LayerNorm>(modules["norm1"]);

        auto norm_hidden_states = norm1->forward(runtime, hidden_states);
        norm_hidden_states = (1.0f + scale_msa) * norm_hidden_states + shift_msa;

        // Conditioning txt stream
        auto norm1_context = std::static_pointer_cast<LayerNorm>(modules["norm1_context"]);

        auto norm_encoder_hidden_states = norm1_context->forward(runtime, encoder_hidden_states);
        norm_encoder_hidden_states = (1.0f + c_scale_msa) * norm_encoder_hidden_states + c_shift_msa;
      
        // Attention on concatenated img + txt stream
        auto attn = std::static_pointer_cast<Flux2Attention<AttnOp>>(modules["attn"]);

        auto [attn_output, context_attn_output] = attn->forward(
            runtime,
            norm_hidden_states,
            norm_encoder_hidden_states,
            std::nullopt,
            image_rotary_emb
        );

        // Process attention outputs for the image stream (`hidden_states`).
        attn_output = gate_msa * attn_output;
        hidden_states = hidden_states + attn_output;

        auto norm2 = std::static_pointer_cast<LayerNorm>(modules["norm2"]);

        norm_hidden_states = norm2->forward(runtime, hidden_states);
        norm_hidden_states = norm_hidden_states * (1.0f + scale_mlp) + shift_mlp;

        auto ff = std::static_pointer_cast<Flux2FeedForward>(modules["ff"]);

        auto ff_output = ff->forward(runtime, norm_hidden_states);
        hidden_states = hidden_states + gate_mlp * ff_output;

        // Process attention outputs for the text stream (`encoder_hidden_states`).
        context_attn_output = c_gate_msa * context_attn_output.value();
        encoder_hidden_states = encoder_hidden_states + context_attn_output.value();

        auto norm2_context = std::static_pointer_cast<LayerNorm>(modules["norm2_context"]);

        norm_encoder_hidden_states = norm2_context->forward(runtime, encoder_hidden_states);
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1.0f + c_scale_mlp) + c_shift_mlp;
        
        auto ff_context = std::static_pointer_cast<Flux2FeedForward>(modules["ff_context"]);

        auto context_ff_output = ff_context->forward(runtime, norm_encoder_hidden_states);
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * context_ff_output;

        if (encoder_hidden_states.dtype() == GGML_TYPE_F16)
            encoder_hidden_states = encoder_hidden_states.clamp(-65504, 65504);

        return {encoder_hidden_states, hidden_states};
    }
};
