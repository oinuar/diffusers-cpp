#include "diffusers/models/resnet/ResnetBlock2D.hpp"

#include "nn/modules/normalization/GroupNorm.hpp"
#include "nn/modules/conv/Conv2d.hpp"
#include "nn/SiLU.hpp"
#include "nn/Linear.hpp"

ResnetBlock2D::ResnetBlock2D(
    int64_t in_channels,
    std::optional<int64_t> out_channels,
    std::optional<int64_t> conv_shortcut,
    float dropout,
    std::optional<int64_t> temb_channels,
    int64_t groups,
    std::optional<int64_t> groups_out,
    float eps,
    const std::string& non_linearity,
    const std::string& time_embedding_norm,
    int64_t kernel,
    std::optional<int64_t> output_scale_factor,
    bool use_in_shortcut,
    bool up,
    bool down,
    bool conv_shortcut_bias,
    int64_t conv_2d_out_channels
)
    : output_scale_factor_(output_scale_factor.value_or(1.0f))
{
    const int64_t out_channels_ = out_channels.value_or(in_channels);
    const int64_t groups_out_ = groups_out.value_or(groups);
    const int64_t conv_2d_out_channels_ =
        conv_2d_out_channels == 0 ? out_channels_ : conv_2d_out_channels;

    use_temb_ = temb_channels.has_value();

    use_shortcut_ = use_in_shortcut || (in_channels != conv_2d_out_channels_);

    modules["norm1"] = std::make_shared<GroupNorm>(
        groups,
        in_channels,
        eps
    );

    // TODO: support other activation functions.
    modules["nonlinearity"] = std::make_shared<SiLU>();

    modules["conv1"] = std::make_shared<Conv2d>(
        in_channels,
        out_channels_,
        kernel,
        1,
        kernel / 2
    );

    if (use_temb_) {
        modules["time_emb_proj"] = std::make_shared<Linear>(
            temb_channels.value(),
            time_embedding_norm == "default"
                ? out_channels_
                : out_channels_ * 2
        );
    }

    modules["norm2"] = std::make_shared<GroupNorm>(
        groups_out_,
        out_channels_,
        eps
    );

    // TODO: Dropout module.
    // modules["dropout"] = std::make_shared<Dropout>(dropout);

    modules["conv2"] = std::make_shared<Conv2d>(
        out_channels_,
        conv_2d_out_channels_,
        kernel,
        1,
        kernel / 2
    );

    if (use_shortcut_) {
        modules["conv_shortcut"] = std::make_shared<Conv2d>(
            in_channels,
            conv_2d_out_channels_,
            1,
            1,
            0,
            conv_shortcut_bias
        );
    }

    // Not needed for AutoencoderKLFlux2 but keep API-compatible.
    if (up) {
        // TODO: Upsample2D
    }

    if (down) {
        // TODO: Downsample2D
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