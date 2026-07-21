#include "diffusers/models/unets/unet2d/UNetMidBlock2D.hpp"

#include "diffusers/models/resnet/ResnetBlock2D.hpp"
#include "nn/attention/Attention.hpp"


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
    /*
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(...),
                ResnetBlock2D(...)
            ]
        )
    */

    modules["resnets.0"] =
        std::make_shared<ResnetBlock2D>(
            in_channels,
            in_channels,
            temb_channels,
            resnet_eps,
            resnet_act_fn,
            output_scale_factor,
            resnet_time_scale_shift,
            resnet_groups
        );


    if (add_attention) {

        modules["attentions.0"] =
            std::make_shared<Attention>(
                in_channels,
                attention_head_dim
            );
    }


    modules["resnets.1"] =
        std::make_shared<ResnetBlock2D>(
            in_channels,
            in_channels,
            temb_channels,
            resnet_eps,
            resnet_act_fn,
            output_scale_factor,
            resnet_time_scale_shift,
            resnet_groups
        );
}



Tensor UNetMidBlock2D::forward(
    ggml_context* ctx,
    Tensor sample,
    std::optional<Tensor> temb
)
{
    /*
        hidden_states = self.resnets[0](hidden_states, temb)
    */

    sample =
        std::static_pointer_cast<ResnetBlock2D>(
            modules["resnets.0"])
        ->forward(
            ctx,
            sample,
            temb
        );


    /*
        hidden_states = self.attentions[0](hidden_states)
    */

    if (add_attention_) {

        sample =
            std::static_pointer_cast<Attention>(
                modules["attentions.0"])
            ->forward(
                ctx,
                sample
            );
    }


    /*
        hidden_states = self.resnets[1](hidden_states, temb)
    */

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