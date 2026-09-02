#include "diffusers/models/unets/unet2d/UNetMidBlock2D.hpp"
#include "nn/SiLU.hpp"
#include "nn/ModuleList.hpp"
#include "diffusers/models/resnet/ResnetBlock2D.hpp"
#include "diffusers/models/attention_processor/Attention.hpp"
#include "nn/attention/ScaledDotProductAttention.hpp"
#include "nn/attention/FlashAttentionOp.hpp"
#include <iostream>

UNetMidBlock2D::UNetMidBlock2D(
    int64_t in_channels,
    std::optional<int64_t> temb_channels,
    float dropout,
    int64_t num_layers,
    float resnet_eps,
    int64_t resnet_groups,
    std::optional<int64_t> attn_groups,
    bool resnet_pre_norm,
    bool add_attention,
    int64_t attention_head_dim,
    float output_scale_factor
) : add_attention_(add_attention)
{
    // there is always at least one resnet
    auto resnets = std::make_shared<ModuleList>(num_layers + 1);
    modules["resnets"] = resnets;

    auto attentions = std::make_shared<ModuleList>(num_layers);
    modules["attentions"] = attentions;

    for (auto i = 0; i < resnets->size(); ++i) {
        if (i > 0 && add_attention) {
            auto heads = in_channels / attention_head_dim;

            (*attentions)[i - 1] =
                std::make_shared<Attention<ScaledDotProductAttention<FlashAttentionOp>>>(
                    in_channels, // query_dim
                    in_channels / attention_head_dim, // heads
                    attention_head_dim, // dim_head
                    0.0f,       // dropout
                    true,        // bias
                    true,        // residual_connection
                    resnet_groups, // norm_num_groups
                    resnet_eps,   // 1e-6
                    output_scale_factor, // 1.0
                    true          // upcast_softmax
                );
        }

        //if resnet_time_scale_shift == "spatial":
        //else:
        (*resnets)[i] = std::make_shared<ResnetBlock2D<SiLU>>(
            in_channels,
            in_channels, // out_channels
            std::nullopt, // conv_shortcut
            0.0f, // dropout
            temb_channels,
            resnet_groups, // groups
            std::nullopt, // groups_out
            true, // per_norm
            resnet_eps, // eps
            false, // skip_time_act
            3, // kernel_size
            output_scale_factor,
            std::nullopt, // use_in_shortcut
            false, // up
            false, // down
            true, // conv_shortcut_bias
            std::nullopt // conv_2d_out_channels
        );
    }
}

Tensor UNetMidBlock2D::forward(Scope scope, Tensor hidden_states, std::optional<Tensor> temb) {
    auto resnets = std::static_pointer_cast<ModuleList>(modules["resnets"]);

    for (auto i = 0; i < resnets->size(); ++i) {
        if (i > 0 && add_attention_) {
            auto attentions = std::static_pointer_cast<ModuleList>(modules["attentions"]);

            hidden_states = std::static_pointer_cast<Attention<ScaledDotProductAttention<FlashAttentionOp>>>((*attentions)[i - 1])
                ->forward(scope, hidden_states);
        }

        hidden_states = std::static_pointer_cast<ResnetBlock2D<SiLU>>((*resnets)[i])->forward(scope, hidden_states, temb);
    }

    return hidden_states;
}
