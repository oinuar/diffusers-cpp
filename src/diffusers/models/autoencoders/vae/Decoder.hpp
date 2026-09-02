#pragma once

#include <optional>
#include <string>
#include <vector>

#include "nn/Module.hpp"
#include "ggml/Context.hpp"
#include "nn/ModuleList.hpp"
#include "nn/SiLU.hpp"
#include "nn/modules/conv/Conv2d.hpp"
#include "nn/modules/normalization/GroupNorm.hpp"
#include "diffusers/models/normalization/SpatialNorm.hpp"
#include "diffusers/models/unets/unet2d/UNetMidBlock2D.hpp"
#include "diffusers/models/unets/unet2d/UpDecoderBlock2D.hpp"

class Decoder : public Module {
public:
    Decoder(
        int64_t in_channels = 3,
        int64_t out_channels = 3,
        const std::vector<int64_t>& block_out_channels = {64},
        int layers_per_block = 2,
        int norm_num_groups = 32,
        bool mid_block_add_attention = true
    )
        : layers_per_block(layers_per_block)
    {
        modules["conv_in"] =
            std::make_shared<Conv2d>(
                in_channels,
                block_out_channels.back(),
                3,
                1,
                1
            );

        modules["mid_block"] =
            std::make_shared<UNetMidBlock2D>(
                block_out_channels.back(), // in_channels,
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

        std::vector<int64_t> reversed_block_out_channels(
            block_out_channels.rbegin(),
            block_out_channels.rend()
        );

        int64_t output_channel = reversed_block_out_channels[0];

        auto up_blocks = std::make_shared<ModuleList>(reversed_block_out_channels.size());
        modules["up_blocks"] = up_blocks;

        for (size_t i = 0; i < reversed_block_out_channels.size(); ++i) {

            int64_t prev_output_channel = output_channel;
            output_channel = reversed_block_out_channels[i];

            bool is_final_block =
                i == reversed_block_out_channels.size() - 1;

            (*up_blocks)[i] =
                std::make_shared<UpDecoderBlock2D>(
                    prev_output_channel, // in_channels
                    output_channel, // out_channels
                    std::nullopt, // resolution_idx
                    0.0f, // dropout
                    layers_per_block + 1, // num_layers
                    1e-6, // resnet_eps
                    norm_num_groups, // resnet_groups
                    true, // resnet_pre_norm
                    1.0f, // output_scale_factor
                    !is_final_block, // add_upsample
                    std::nullopt // temb_channels
                );
        }

        /*
        if norm_type == "spatial":
            self.conv_norm_out = SpatialNorm(...)
        else:
            self.conv_norm_out = GroupNorm(...)
        */

        modules["conv_norm_out"] =
            std::make_shared<GroupNorm>(
                norm_num_groups, // num_groups
                block_out_channels[0], // num_channels
                1e-6 // eps
            );

        modules["conv_act"] =
            std::make_shared<SiLU>();

        modules["conv_out"] =
            std::make_shared<Conv2d>(
                block_out_channels[0], // in_channels
                out_channels, // out_channels
                3, // kernel_size
                1, // stride
                1 // padding
            );
    }

    Tensor forward(Scope scope, Tensor sample, std::optional<Tensor> latent_embeds = std::nullopt) {
        sample =
            std::static_pointer_cast<Conv2d>(
                modules["conv_in"])
            ->forward(scope, sample);

        sample =
            std::static_pointer_cast<UNetMidBlock2D>(
                modules["mid_block"])
            ->forward(
                scope,
                sample,
                latent_embeds
            );

        auto up_blocks = std::static_pointer_cast<ModuleList>(modules["up_blocks"]);

        for (auto i = 0; i < up_blocks->size(); ++i) {
            auto up_block = std::static_pointer_cast<UpDecoderBlock2D>((*up_blocks)[i]);

            sample = up_block->forward(scope, sample, latent_embeds);
        }

        if (latent_embeds) {
            sample =
                std::static_pointer_cast<SpatialNorm>(
                    modules["conv_norm_out"])
                ->forward(
                    scope,
                    sample,
                    latent_embeds.value()
                );
        }
        else {
            sample =
                std::static_pointer_cast<GroupNorm>(
                    modules["conv_norm_out"])
                ->forward(
                    scope,
                    sample
                );
        }

        sample =
            std::static_pointer_cast<SiLU>(
                modules["conv_act"])
            ->forward(scope, sample);


        sample =
            std::static_pointer_cast<Conv2d>(
                modules["conv_out"])
            ->forward(scope, sample);

        return sample;
    }


private:
    int layers_per_block;
};