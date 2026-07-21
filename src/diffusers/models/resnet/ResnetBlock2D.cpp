#include "diffusers/models/resnet/ResnetBlock2D.hpp"

#include "nn/modules/normalization/GroupNorm.hpp"
#include "nn/modules/conv/Conv2d.hpp"
#include "nn/SiLU.hpp"
#include "nn/Linear.hpp"


ResnetBlock2D::ResnetBlock2D(
    int64_t in_channels,
    int64_t out_channels,
    std::optional<int64_t> temb_channels,
    float eps,
    const std::string& act_fn,
    float output_scale_factor,
    const std::string& time_scale_shift,
    int64_t groups
)
    : output_scale_factor_(output_scale_factor),
      use_shortcut_(in_channels != out_channels),
      use_temb_(temb_channels.has_value())
{
    /*
        self.norm1 = GroupNorm(
            groups,
            in_channels,
            eps
        )
    */
    modules["norm1"] =
        std::make_shared<GroupNorm>(
            groups,
            in_channels,
            eps
        );


    modules["act"] =
        std::make_shared<SiLU>();


    /*
        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            3,
            padding=1
        )
    */
    modules["conv1"] =
        std::make_shared<Conv2d>(
            in_channels,
            out_channels,
            3,
            1,
            1
        );


    /*
        temb projection
        only needed when temporal embedding exists
    */

    if (use_temb_) {
        modules["time_emb_proj"] =
            std::make_shared<Linear>(
                temb_channels.value(),
                out_channels
            );
    }


    /*
        norm2
    */

    modules["norm2"] =
        std::make_shared<GroupNorm>(
            groups,
            out_channels,
            eps
        );


    /*
        conv2
    */

    modules["conv2"] =
        std::make_shared<Conv2d>(
            out_channels,
            out_channels,
            3,
            1,
            1
        );


    /*
        shortcut
    */

    if (use_shortcut_) {
        modules["conv_shortcut"] =
            std::make_shared<Conv2d>(
                in_channels,
                out_channels,
                1,
                1,
                0
            );
    }
}

Tensor ResnetBlock2D::forward(
    ggml_context* ctx,
    Tensor hidden_states,
    std::optional<Tensor> temb
)
{
    auto residual = hidden_states;


    /*
        hidden_states = norm1(hidden_states)
    */

    hidden_states =
        std::static_pointer_cast<GroupNorm>(
            modules["norm1"])
        ->forward(
            ctx,
            hidden_states
        );


    hidden_states =
        std::static_pointer_cast<SiLU>(
            modules["act"])
        ->forward(
            ctx,
            hidden_states
        );


    hidden_states =
        std::static_pointer_cast<Conv2d>(
            modules["conv1"])
        ->forward(
            ctx,
            hidden_states
        );


    /*
        temb conditioning

        Diffusers:

        temb = self.time_emb_proj(self.nonlinearity(temb))
        hidden_states = hidden_states + temb[:, :, None, None]
    */

    if (use_temb_ && temb) {

        auto t =
            std::static_pointer_cast<Linear>(
                modules["time_emb_proj"])
            ->forward(
                ctx,
                std::static_pointer_cast<SiLU>(
                    modules["act"])
                ->forward(
                    ctx,
                    temb.value()
                )
            );


        hidden_states =
            hidden_states + t;
    }


    /*
        norm2
    */

    hidden_states =
        std::static_pointer_cast<GroupNorm>(
            modules["norm2"])
        ->forward(
            ctx,
            hidden_states
        );


    hidden_states =
        std::static_pointer_cast<SiLU>(
            modules["act"])
        ->forward(
            ctx,
            hidden_states
        );


    /*
        conv2
    */

    hidden_states =
        std::static_pointer_cast<Conv2d>(
            modules["conv2"])
        ->forward(
            ctx,
            hidden_states
        );


    /*
        shortcut
    */

    if (use_shortcut_) {

        residual =
            std::static_pointer_cast<Conv2d>(
                modules["conv_shortcut"])
            ->forward(
                ctx,
                residual
            );
    }


    hidden_states =
        hidden_states + residual;


    hidden_states =
        hidden_states / output_scale_factor_;


    return hidden_states;
}