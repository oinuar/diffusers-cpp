#pragma once

#include <optional>
#include <string>
#include <vector>

#include "nn/Module.hpp"
#include "nn/SiLU.hpp"
#include "nn/modules/conv/Conv2d.hpp"
#include "nn/modules/normalization/GroupNorm.hpp"
#include "nn/modules/normalization/SpatialNorm.hpp"
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
        const std::string& act_fn = "silu",
        const std::string& norm_type = "group",
        bool mid_block_add_attention = true
    )
        : layers_per_block(layers_per_block),
          norm_type(norm_type)
    {
        /*
         self.conv_in = nn.Conv2d(
             in_channels,
             block_out_channels[-1],
             kernel_size=3,
             stride=1,
             padding=1,
         )
        */
        modules["conv_in"] =
            std::make_shared<Conv2d>(
                in_channels,
                block_out_channels.back(),
                3,
                1,
                1
            );


        /*
        temb_channels = in_channels if norm_type == "spatial" else None
        */
        std::optional<int64_t> temb_channels;

        if (norm_type == "spatial")
            temb_channels = in_channels;


        /*
        self.mid_block = UNetMidBlock2D(...)
        */
        modules["mid_block"] =
            std::make_shared<UNetMidBlock2D>(
                block_out_channels.back(),
                1e-6,
                act_fn,
                1.0f,
                norm_type == "group" ? "default" : norm_type,
                block_out_channels.back(),
                norm_num_groups,
                temb_channels,
                mid_block_add_attention
            );


        /*
        reversed_block_out_channels = reversed(block_out_channels)
        */
        std::vector<int64_t> reversed_channels(
            block_out_channels.rbegin(),
            block_out_channels.rend()
        );


        /*
        output_channel = reversed_block_out_channels[0]
        */
        int64_t output_channel = reversed_channels[0];


        /*
        for i, up_block_type in enumerate(up_block_types):
        */
        for (size_t i = 0; i < reversed_channels.size(); ++i) {

            int64_t prev_output_channel = output_channel;
            output_channel = reversed_channels[i];


            /*
            is_final_block = i == len(block_out_channels)-1
            */
            bool is_final_block =
                i == reversed_channels.size() - 1;


            /*
            up_block = get_up_block(...)
            */
            modules["up_blocks." + std::to_string(i)] =
                std::make_shared<UpDecoderBlock2D>(
                    layers_per_block + 1,
                    prev_output_channel,
                    output_channel,
                    prev_output_channel,
                    !is_final_block,
                    1e-6,
                    act_fn,
                    norm_num_groups,
                    output_channel,
                    temb_channels,
                    norm_type
                );
        }


        /*
        if norm_type == "spatial":
            self.conv_norm_out = SpatialNorm(...)
        else:
            self.conv_norm_out = GroupNorm(...)
        */

        if (norm_type == "spatial") {
            modules["conv_norm_out"] =
                std::make_shared<SpatialNorm>(
                    block_out_channels[0],
                    temb_channels.value()
                );
        }
        else {
            modules["conv_norm_out"] =
                std::make_shared<GroupNorm>(
                    norm_num_groups,
                    block_out_channels[0],
                    1e-6
                );
        }


        /*
        self.conv_act = nn.SiLU()
        */
        modules["conv_act"] =
            std::make_shared<SiLU>();


        /*
        self.conv_out = nn.Conv2d(...)
        */
        modules["conv_out"] =
            std::make_shared<Conv2d>(
                block_out_channels[0],
                out_channels,
                3,
                1,
                1
            );
    }


    Tensor forward(
        ggml_context* ctx,
        Tensor sample,
        std::optional<Tensor> latent_embeds = std::nullopt
    ) {
        /*
        sample = self.conv_in(sample)
        */
        sample =
            std::static_pointer_cast<Conv2d>(
                modules["conv_in"])
            ->forward(ctx, sample);


        /*
        sample = self.mid_block(sample, latent_embeds)
        */
        sample =
            std::static_pointer_cast<UNetMidBlock2D>(
                modules["mid_block"])
            ->forward(
                ctx,
                sample,
                latent_embeds
            );


        /*
        for up_block in self.up_blocks:
            sample = up_block(sample, latent_embeds)
        */
        for (int i = 0; ; ++i) {

            auto key =
                "up_blocks." + std::to_string(i);

            auto it = modules.find(key);

            if (it == modules.end())
                break;

            sample =
                std::static_pointer_cast<UpDecoderBlock2D>(
                    it->second)
                ->forward(
                    ctx,
                    sample,
                    latent_embeds
                );
        }


        /*
        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)
        */

        if (latent_embeds) {
            sample =
                std::static_pointer_cast<SpatialNorm>(
                    modules["conv_norm_out"])
                ->forward(
                    ctx,
                    sample,
                    latent_embeds.value()
                );
        }
        else {
            sample =
                std::static_pointer_cast<GroupNorm>(
                    modules["conv_norm_out"])
                ->forward(
                    ctx,
                    sample
                );
        }


        /*
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        */

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
    std::string norm_type;
};