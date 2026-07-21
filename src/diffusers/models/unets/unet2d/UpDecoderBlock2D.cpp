#include "diffusers/models/unets/unet2d/UpDecoderBlock2D.hpp"

#include "diffusers/models/resnet/ResnetBlock2D.hpp"
#include "diffusers/models/upsampling/Upsample2D.hpp"


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

        /*
            First resnet receives previous block channels.
            Remaining resnets receive out_channels.
        */

        int64_t resnet_in_channels =
            i == 0 ? in_channels : out_channels;


        modules["resnets." + std::to_string(i)] =
            std::make_shared<ResnetBlock2D>(
                resnet_in_channels,
                out_channels,
                temb_channels,
                resnet_eps,
                resnet_act_fn,
                1.0f,
                resnet_time_scale_shift,
                resnet_groups
            );
    }


    if (add_upsample_) {
        modules["upsamplers.0"] =
            std::make_shared<Upsample2D>(
                out_channels
            );
    }
}