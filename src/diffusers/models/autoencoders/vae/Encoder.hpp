#pragma once

#include "nn/Module.hpp"
#include "nn/ModuleList.hpp"
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
        int64_t layers_per_block = 2,
        int64_t norm_num_groups = 32,
        bool double_z = true,
        bool mid_block_add_attention = true
    ) {
        modules["conv_in"] =
            std::make_shared<Conv2d>(
                in_channels,
                block_out_channels[0], // out_channels
                3, // kernel_size
                1, // stride
                1 // padding
            );

        auto down_blocks = std::make_shared<ModuleList>(block_out_channels.size());
        modules["down_blocks"] = down_blocks;

        auto output_channel = block_out_channels[0];

        for (auto i = 0; i < block_out_channels.size(); ++i) {
            auto input_channel = output_channel;
            output_channel = block_out_channels[i];

            bool is_final_block = i == block_out_channels.size() - 1;

            (*down_blocks)[i] = 
                std::make_shared<DownEncoderBlock2D>(
                    input_channel, // in_channels
                    output_channel, // out_channels
                    0.0f, // dropout
                    layers_per_block, // num_layers
                    1e-6, // resnet_eps
                    norm_num_groups, // resnet_groups
                    true, // resnet_pre_norm
                    1.0f, // output_scale_factor
                    !is_final_block, // add_downsample
                    1 // downsample_padding
                );
        }

        modules["mid_block"] =
            std::make_shared<UNetMidBlock2D>(
                block_out_channels.back(), // in_channels
                std::nullopt, // temb_channels
                0.0f, // dropout
                1, // num_layers
                1e-6, // resnet_eps
                norm_num_groups, // resnet_groups
                std::nullopt, // attn_groups
                true, // resnet_pre_norm
                mid_block_add_attention, // add_attention
                block_out_channels.back(), // attention_head_dim
                1.0f // output_scale_factor
            );

        modules["conv_norm_out"] =
            std::make_shared<GroupNorm>(
                norm_num_groups, // num_groups
                block_out_channels.back(), // num_channels
                1e-6 // eps
            );

        modules["conv_act"] = std::make_shared<SiLU>();

        auto conv_out_channels = double_z ? 2 * out_channels : out_channels;

        modules["conv_out"] =
            std::make_shared<Conv2d>(
                block_out_channels.back(), // in_channels
                conv_out_channels, // out_channels
                3, // kernel_size
                1, // stride
                1 // padding
            );
    }

    Tensor forward(ggml_context* ctx, Tensor sample) {
        auto conv_in = std::static_pointer_cast<Conv2d>(modules["conv_in"]);
        sample = conv_in->forward(ctx, sample);
        
        auto down_blocks = std::static_pointer_cast<ModuleList>(modules["down_blocks"]);

        // down
        for (auto i = 0; i < down_blocks->size(); ++i) {
            auto block = std::static_pointer_cast<DownEncoderBlock2D>((*down_blocks)[i]);
            sample = block->forward(ctx, sample);
        }

        // middle
        auto mid_block = std::static_pointer_cast<UNetMidBlock2D>(modules["mid_block"]);
        sample = mid_block->forward(ctx, sample);

        // post-process
        auto conv_norm_out = std::static_pointer_cast<GroupNorm>(modules["conv_norm_out"]);
        sample = conv_norm_out->forward(ctx, sample);

        auto conv_act = std::static_pointer_cast<SiLU>(modules["conv_act"]);
        sample = conv_act->forward(ctx, sample);

        auto conv_out = std::static_pointer_cast<Conv2d>(modules["conv_out"]);
        sample = conv_out->forward(ctx, sample);

        return sample;
    }

private:
    int layers_per_block;
};