#include "diffusers/models/unets/unet2d/UpDecoderBlock2D.hpp"

#include "diffusers/models/resnet/ResnetBlock2D.hpp"
#include "diffusers/models/resnet/Upsample2D.hpp"

UpDecoderBlock2D::UpDecoderBlock2D(
    int64_t num_layers,
    int64_t in_channels,
    int64_t out_channels,
    int64_t prev_output_channel,
    bool add_upsample,
    float resnet_eps,
    const std::string& resnet_act_fn,
    int64_t resnet_groups,
    int64_t attention_head_dim,
    std::optional<int64_t> temb_channels,
    const std::string& resnet_time_scale_shift
)
    : add_upsample_(add_upsample),
      num_layers_(num_layers)
{
    for (int64_t i = 0; i < num_layers; ++i) {

        const int64_t resnet_in_channels =
            i == 0 ? prev_output_channel : out_channels;


        modules["resnets." + std::to_string(i)] =
            std::make_shared<ResnetBlock2D>(
                /* in_channels           */ resnet_in_channels,
                /* out_channels          */ out_channels,
                /* conv_shortcut         */ std::nullopt,
                /* dropout               */ 0.0f,
                /* temb_channels         */ temb_channels,
                /* groups                */ resnet_groups,
                /* groups_out            */ resnet_groups,
                /* eps                   */ resnet_eps,
                /* non_linearity         */ resnet_act_fn,
                /* time_embedding_norm   */ resnet_time_scale_shift,
                /* kernel                */ 3,
                /* output_scale_factor   */ 1.0f,
                /* use_in_shortcut       */ false,
                /* up                    */ false,
                /* down                  */ false,
                /* conv_shortcut_bias    */ true,
                /* conv_2d_out_channels  */ 0
            );
    }


    if (add_upsample_) {
        modules["upsamplers.0"] =
            std::make_shared<Upsample2D>(
                out_channels,
                true,
                out_channels
            );
    }
}

Tensor UpDecoderBlock2D::forward(
    ggml_context* ctx,
    Tensor hidden_states,
    std::optional<Tensor> temb
)
{
    for (int64_t i = 0; i < num_layers_; ++i) {

        hidden_states =
            std::static_pointer_cast<ResnetBlock2D>(
                modules["resnets." + std::to_string(i)])
            ->forward(
                ctx,
                hidden_states,
                temb
            );
    }


    if (add_upsample_) {

        hidden_states =
            std::static_pointer_cast<Upsample2D>(
                modules["upsamplers.0"])
            ->forward(
                ctx,
                hidden_states
            );
    }


    return hidden_states;
}