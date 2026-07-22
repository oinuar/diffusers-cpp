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
    const int64_t out_channels_ =
        out_channels.value_or(in_channels);

    const int64_t groups_out_ =
        groups_out.value_or(groups);

    const int64_t conv_2d_out_channels_ =
        conv_2d_out_channels == 0
            ? out_channels_
            : conv_2d_out_channels;


    /*
        Diffusers:

        self.use_in_shortcut = self.in_channels != self.out_channels
        self.use_shortcut = use_in_shortcut or self.use_in_shortcut

        In C++ we only need the effective flag because
        conv_shortcut is created and used together.
    */
    use_in_shortcut_ =
        use_in_shortcut ||
        (in_channels != out_channels_);


    use_temb_ =
        temb_channels.has_value();


    modules["norm1"] =
        std::make_shared<GroupNorm>(
            groups,
            in_channels,
            eps
        );


    modules["nonlinearity"] =
        std::make_shared<SiLU>();


    modules["conv1"] =
        std::make_shared<Conv2d>(
            in_channels,
            out_channels_,
            kernel,
            1,
            kernel / 2
        );


    if (use_temb_) {

        modules["time_emb_proj"] =
            std::make_shared<Linear>(
                temb_channels.value(),
                time_embedding_norm == "default"
                    ? out_channels_
                    : out_channels_ * 2
            );
    }


    modules["norm2"] =
        std::make_shared<GroupNorm>(
            groups_out_,
            out_channels_,
            eps
        );


    modules["conv2"] =
        std::make_shared<Conv2d>(
            out_channels_,
            conv_2d_out_channels_,
            kernel,
            1,
            kernel / 2
        );


    /*
        Diffusers creates conv_shortcut when
        input channels != output channels.
    */
    if (use_in_shortcut_) {

        modules["conv_shortcut"] =
            std::make_shared<Conv2d>(
                in_channels,
                conv_2d_out_channels_,
                1,
                1,
                0,
                conv_shortcut_bias
            );
    }


    if (up) {
        // TODO: Upsample2D
    }

    if (down) {
        // TODO: Downsample2D
    }
}


Tensor ResnetBlock2D::forward(
    ggml_context* ctx,
    Tensor input_tensor,
    std::optional<Tensor> temb
)
{
    Tensor hidden_states = input_tensor;


    hidden_states =
        std::static_pointer_cast<GroupNorm>(
            modules["norm1"])
        ->forward(
            ctx,
            hidden_states
        );


    hidden_states =
        std::static_pointer_cast<SiLU>(
            modules["nonlinearity"])
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


    if (use_temb_ && temb) {

        auto t =
            std::static_pointer_cast<SiLU>(
                modules["nonlinearity"])
            ->forward(
                ctx,
                *temb
            );


        t =
            std::static_pointer_cast<Linear>(
                modules["time_emb_proj"])
            ->forward(
                ctx,
                t
            );


        /*
            PyTorch:

                temb[:, :, None, None]

            C++:

                [N,C] -> [N,C,1,1]
        */
        t =
            t[{
                Tensor::Slice::all(),
                Tensor::Slice::all(),
                Tensor::Slice::none(),
                Tensor::Slice::none()
            }];


        hidden_states =
            hidden_states + t;
    }


    hidden_states =
        std::static_pointer_cast<GroupNorm>(
            modules["norm2"])
        ->forward(
            ctx,
            hidden_states
        );


    hidden_states =
        std::static_pointer_cast<SiLU>(
            modules["nonlinearity"])
        ->forward(
            ctx,
            hidden_states
        );


    hidden_states =
        std::static_pointer_cast<Conv2d>(
            modules["conv2"])
        ->forward(
            ctx,
            hidden_states
        );


    /*
        Residual projection.

        For channel projection:

            input  [N,Cin,H,W]
            output [N,Cout,H,W]

        conv_shortcut performs Cin -> Cout.
    */
    if (use_in_shortcut_) {

        input_tensor =
            std::static_pointer_cast<Conv2d>(
                modules["conv_shortcut"])
            ->forward(
                ctx,
                input_tensor
            );
    }


    return (input_tensor + hidden_states) / output_scale_factor_;
}
