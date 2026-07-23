#include "diffusers/models/unets/unet2d/UNetMidBlock2D.hpp"

#include "diffusers/models/resnet/ResnetBlock2D.hpp"
#include "diffusers/models/attention_processor/Attention.hpp"
#include "nn/attention/ScaledDotProductAttention.hpp"
#include "nn/attention/FlashAttentionOp.hpp"

UNetMidBlock2D::UNetMidBlock2D(
    int64_t in_channels,
    float resnet_eps,
    const std::string& resnet_act_fn,
    float output_scale_factor,
    const std::string& resnet_time_scale_shift,
    int64_t attention_head_dim,
    int64_t resnet_groups,
    std::optional<int64_t> temb_channels,
    bool add_attention
)
    : add_attention_(add_attention)
{
    modules["resnets.0"] =
        std::make_shared<ResnetBlock2D>(
            /* in_channels           */ in_channels,
            /* out_channels          */ in_channels,
            /* conv_shortcut         */ std::nullopt,
            /* dropout               */ 0.0f,
            /* temb_channels         */ temb_channels,
            /* groups                */ resnet_groups,
            /* groups_out            */ std::nullopt,
            /* eps                   */ resnet_eps,
            /* non_linearity         */ resnet_act_fn,
            /* time_embedding_norm   */ resnet_time_scale_shift,
            /* kernel                */ 3,
            /* output_scale_factor   */ output_scale_factor,
            /* use_in_shortcut       */ false,
            /* up                    */ false,
            /* down                  */ false,
            /* conv_shortcut_bias    */ true,
            /* conv_2d_out_channels  */ 0
        );

    if (add_attention) {
        int64_t heads = in_channels / attention_head_dim;

        modules["attentions.0"] =
            std::make_shared<Attention<ScaledDotProductAttention<FlashAttentionOp>>>(
                in_channels,
                in_channels / attention_head_dim,
                attention_head_dim,

                0.0f,       // dropout

                true,        // qkv bias
                true,        // residual_connection

                resnet_groups, // norm_num_groups = attn_groups

                resnet_eps,   // 1e-6

                output_scale_factor, // 1.0

                true          // upcast_softmax
            );
    }

    modules["resnets.1"] =
        std::make_shared<ResnetBlock2D>(
            /* in_channels           */ in_channels,
            /* out_channels          */ in_channels,
            /* conv_shortcut         */ std::nullopt,
            /* dropout               */ 0.0f,
            /* temb_channels         */ temb_channels,
            /* groups                */ resnet_groups,
            /* groups_out            */ std::nullopt,
            /* eps                   */ resnet_eps,
            /* non_linearity         */ resnet_act_fn,
            /* time_embedding_norm   */ resnet_time_scale_shift,
            /* kernel                */ 3,
            /* output_scale_factor   */ output_scale_factor,
            /* use_in_shortcut       */ false,
            /* up                    */ false,
            /* down                  */ false,
            /* conv_shortcut_bias    */ true,
            /* conv_2d_out_channels  */ 0
        );
}

Tensor UNetMidBlock2D::forward(
    ggml_context* ctx,
    Tensor sample,
    std::optional<Tensor> temb
)
{
    // 1. First ResNet block
    sample =
        std::static_pointer_cast<ResnetBlock2D>(
            modules["resnets.0"])
        ->forward(
            ctx,
            sample,
            temb
        );

    // 2. Attention block (if enabled)
    if (add_attention_) {
        sample =
            std::static_pointer_cast<
                Attention<ScaledDotProductAttention<FlashAttentionOp>>
            >(modules["attentions.0"])
            ->forward(
                ctx,
                sample
            );
    }

    // 3. Second ResNet block
    sample =
        std::static_pointer_cast<ResnetBlock2D>(
            modules["resnets.1"])
        ->forward(
            ctx,
            sample,
            temb
        );

    return sample;
}