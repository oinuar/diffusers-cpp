#include "diffusers/models/unets/unet2d/DownEncoderBlock2D.hpp"

#include "diffusers/models/resnet/ResnetBlock2D.hpp"
#include "diffusers/models/downsampling/Downsample2D.hpp"

DownEncoderBlock2D::DownEncoderBlock2D(
    int64_t num_layers,
    int64_t in_channels,
    int64_t out_channels,
    bool add_downsample,
    float resnet_eps,
    const std::string& resnet_act_fn,
    int64_t resnet_groups,
    std::optional<int64_t> temb_channels
)
    : num_layers_(num_layers),
      add_downsample_(add_downsample)
{
    for (int64_t i = 0; i < num_layers; ++i) {
        const int64_t block_in_channels =
            i == 0 ? in_channels : out_channels;

        modules["resnets." + std::to_string(i)] =
            std::make_shared<ResnetBlock2D>(
                /* in_channels            */ block_in_channels,
                /* out_channels           */ out_channels,
                /* conv_shortcut          */ std::nullopt,
                /* dropout               */ 0.0f,
                /* temb_channels         */ temb_channels,
                /* groups                */ resnet_groups,
                /* groups_out            */ std::nullopt,
                /* eps                   */ resnet_eps,
                /* non_linearity         */ resnet_act_fn,
                /* time_embedding_norm   */ "default",
                /* kernel                */ 3,
                /* output_scale_factor   */ 1.0f,
                /* use_in_shortcut       */ false,
                /* up                    */ false,
                /* down                  */ false,
                /* conv_shortcut_bias    */ true,
                /* conv_2d_out_channels  */ 0
            );
    }

    if (add_downsample_) {
        modules["downsamplers.0"] =
            std::make_shared<Downsample2D>(
                out_channels,
                true,   // use_conv
                out_channels,
                1
            );
    }
}

Tensor DownEncoderBlock2D::forward(
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


    if (add_downsample_) {

        hidden_states =
            std::static_pointer_cast<Downsample2D>(
                modules["downsamplers.0"])
            ->forward(
                ctx,
                hidden_states
            );
    }


    return hidden_states;
}