#include "diffusers/models/unets/unet2d/UpDecoderBlock2D.hpp"
#include "nn/ModuleList.hpp"
#include "diffusers/models/resnet/ResnetBlock2D.hpp"
#include "diffusers/models/resnet/Upsample2D.hpp"

UpDecoderBlock2D::UpDecoderBlock2D(
    int64_t in_channels,
    int64_t out_channels,
    std::optional<int64_t> resolution_idx,
    float dropout,
    int64_t num_layers,
    float resnet_eps,
    int64_t resnet_groups,
    bool resnet_pre_norm,
    float output_scale_factor,
    bool add_upsample,
    std::optional<int64_t> temb_channels
) : add_upsample_(add_upsample)
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
            temb_channels,
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

    if (add_upsample)
        modules["upsamplers"] = std::shared_ptr<Module>(new ModuleList({
            std::shared_ptr<Module>(new Upsample2D(
                out_channels, // channels
                true, // use_conv
                out_channels,
                false // use_conv_transpose
            ))
        }));
}

Tensor UpDecoderBlock2D::forward(Runtime& runtime, Tensor hidden_states, std::optional<Tensor> temb) {
    auto resnets = std::static_pointer_cast<ModuleList>(modules["resnets"]);

    for (auto i = 0; i < resnets->size(); ++i)
        hidden_states = std::static_pointer_cast<ResnetBlock2D<SiLU>>((*resnets)[i])->forward(runtime, hidden_states);

    if (add_upsample_) {
        auto upsamplers = std::static_pointer_cast<ModuleList>(modules["upsamplers"]);

        for (auto i = 0; i < upsamplers->size(); ++i)
            hidden_states = std::static_pointer_cast<Upsample2D>((*upsamplers)[i])->forward(runtime, hidden_states);
    }

    return hidden_states;
}
