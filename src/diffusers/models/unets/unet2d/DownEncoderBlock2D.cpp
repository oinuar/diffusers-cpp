#include "diffusers/models/unets/unet2d/DownEncoderBlock2D.hpp"
#include "nn/ModuleList.hpp"
#include "nn/SiLU.hpp"
#include "diffusers/models/resnet/ResnetBlock2D.hpp"
#include "diffusers/models/downsampling/Downsample2D.hpp"

DownEncoderBlock2D::DownEncoderBlock2D(
    int64_t in_channels,
    int64_t out_channels,
    float dropout,
    int64_t num_layers,
    float resnet_eps,
    int64_t resnet_groups,
    bool resnet_pre_norm,
    float output_scale_factor,
    bool add_downsample,
    int64_t downsample_padding 
) : add_downsample_(add_downsample)
{
    auto resnets = std::make_shared<ModuleList>(num_layers);
    modules["resnets"] = resnets;

    for (auto i = 0; i < resnets->size(); ++i) {
        auto input_channels = i == 0 ? in_channels : out_channels;

        // if resnet_time_scale_shift == "spatial":
        // else:
        (*resnets)[i] = std::make_shared<ResnetBlock2D<SiLU>>(
            input_channels, // in_channels
            out_channels,
            std::nullopt, // conv_shortcut
            0.0f, // dropout
            std::nullopt, // temb_channels
            resnet_groups, // groups
            std::nullopt, // groups_out
            resnet_pre_norm, // per_norm
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
    
    if (add_downsample)
        modules["downsamplers"] = std::shared_ptr<Module>(new ModuleList({
            std::shared_ptr<Module>(new Downsample2D(
                out_channels, // channels
                true, // use_conv
                out_channels,
                downsample_padding // padding
            ))
        }));
}

Tensor DownEncoderBlock2D::forward(Context& context, Tensor hidden_states) {
    auto resnets = std::static_pointer_cast<ModuleList>(modules["resnets"]);

    for (auto i = 0; i < resnets->size(); ++i)
        hidden_states = std::static_pointer_cast<ResnetBlock2D<SiLU>>((*resnets)[i])->forward(context, hidden_states);

    if (add_downsample_) {
        auto downsamplers = std::static_pointer_cast<ModuleList>(modules["downsamplers"]);

        for (auto i = 0; i < downsamplers->size(); ++i)
            hidden_states = std::static_pointer_cast<Downsample2D>((*downsamplers)[i])->forward(context, hidden_states);
    }

    return hidden_states;
}
