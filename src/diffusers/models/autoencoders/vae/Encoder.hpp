#pragma once

#include "nn/Module.hpp"
#include "nn/SiLU.hpp"
#include "nn/modules/conv/Conv2d.hpp"
#include "nn/modules/normalization/GroupNorm.hpp"
#include "diffusers/models/unets/unet2d/UNetMidBlock2D.hpp"
#include "diffusers/models/unets/unet2d/DownEncoderBlock2D.hpp"

class Encoder : public Module {
public:
    Encoder(
        int64_t in_channels = 3,
        int64_t out_channels = 3,
        std::vector<int64_t> block_out_channels = {64},
        int layers_per_block = 2,
        int norm_num_groups = 32,
        bool double_z = true,
        bool mid_block_add_attention = true
    ) {
        this->layers_per_block = layers_per_block;

        modules["conv_in"] =
            std::make_shared<Conv2d>(
                in_channels,
                block_out_channels[0],
                3,
                1,
                1
            );

        int64_t output_channel = block_out_channels[0];

        for (size_t i = 0; i < block_out_channels.size(); ++i) {
            int64_t input_channel = output_channel;
            output_channel = block_out_channels[i];

            bool add_downsample =
                i != block_out_channels.size() - 1;

            modules["down_blocks." + std::to_string(i)] =
                std::make_shared<DownEncoderBlock2D>(
                    layers_per_block,
                    input_channel,
                    output_channel,
                    add_downsample,
                    1e-6,
                    norm_num_groups
                );
        }

        modules["mid_block"] =
            std::make_shared<UNetMidBlock2D>(
                block_out_channels.back(),
                1e-6,
                norm_num_groups,
                mid_block_add_attention
            );


        modules["conv_norm_out"] =
            std::make_shared<GroupNorm>(
                norm_num_groups,
                block_out_channels.back(),
                1e-6
            );

        modules["conv_act"] =
            std::make_shared<SiLU>();

        modules["conv_out"] =
            std::make_shared<Conv2d>(
                block_out_channels.back(),
                double_z ? out_channels * 2 : out_channels,
                3,
                1,
                1
            );
    }

    Tensor forward(ggml_context* ctx, Tensor sample) {
        auto conv_in =
            std::static_pointer_cast<Conv2d>(
                modules["conv_in"]);

        sample = conv_in->forward(ctx, sample);


        for (int i = 0; ; ++i) {
            auto it = modules.find(
                "down_blocks." + std::to_string(i)
            );

            if (it == modules.end())
                break;

            auto block =
                std::static_pointer_cast<DownEncoderBlock2D>(
                    it->second);

            sample = block->forward(ctx, sample);
        }


        sample =
            std::static_pointer_cast<UNetMidBlock2D>(
                modules["mid_block"])
            ->forward(ctx, sample);


        sample =
            std::static_pointer_cast<GroupNorm>(
                modules["conv_norm_out"])
            ->forward(ctx, sample);

        sample =
            std::static_pointer_cast<SiLU>(
                modules["conv_act"])
            ->forward(ctx, sample);

        sample =
            std::static_pointer_cast<Conv2d>(
                modules["conv_out"])
            ->forward(ctx, sample);

        return sample;
    }

private:
    int layers_per_block;
};