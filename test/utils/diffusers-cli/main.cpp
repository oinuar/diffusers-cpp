#include "../TestCLI.hpp"
#include "nn/Parameter.hpp"
#include "nn/Visitor.hpp"
#include "nn/RethrowVisitor.hpp"
#include "nn/attention/ScaledDotProductAttention.hpp"
#include "nn/attention/FlashAttentionOp.hpp"

#include "diffusers/models/normalization/AdaLayerNormContinuous.hpp"
#include "diffusers/models/normalization/SpatialNorm.hpp"
#include "diffusers/models/resnet/Upsample2D.hpp"
#include "diffusers/models/resnet/ResnetBlock2D.hpp"
#include "diffusers/models/downsampling/Downsample2D.hpp"
#include "diffusers/models/embeddings/TimestepEmbedding.hpp"
#include "diffusers/models/embeddings/Timesteps.hpp"
#include "diffusers/models/attention_processor/Attention.hpp"

#include "diffusers/models/unets/unet2d/UNetMidBlock2D.hpp"

#include "diffusers/models/transformers/flux2/Flux2SwiGLU.hpp"
#include "diffusers/models/transformers/flux2/Flux2FeedForward.hpp"
#include "diffusers/models/transformers/flux2/Flux2Modulation.hpp"
#include "diffusers/models/transformers/flux2/Flux2TimestepGuidanceEmbeddings.hpp"
#include "diffusers/models/transformers/flux2/Flux2PosEmbed.hpp"
#include "diffusers/models/transformers/flux2/Flux2Attention.hpp"
#include "diffusers/models/transformers/flux2/Flux2ParallelSelfAttention.hpp"
#include "diffusers/models/transformers/flux2/Flux2SingleTransformerBlock.hpp"
#include "diffusers/models/transformers/flux2/Flux2TransformerBlock.hpp"
#include "diffusers/models/transformers/flux2/Flux2Transformer2DModel.hpp"

#include <numeric>

class TestDiffusersCLI : public TestCLI {
public:
    TestDiffusersCLI(int argc, char** argv) : TestCLI(argc, argv) {}

    virtual Plan build(Context& ctx) {

        if (args_.get(0) == "AdaLayerNormContinuous") {
            auto embedding_dim = args_.get_one<int64_t>("--embedding_dim");
            auto conditioning_embedding_dim = args_.get_one<int64_t>("--conditioning_embedding_dim");
            auto elementwise_affine = args_.get_optional<bool>("--elementwise_affine").value_or(true);
            auto eps = args_.get_optional<float>("--eps").value_or(1e-5f);
            auto bias = args_.get_optional<bool>("--bias").value_or(true);
            auto hidden_states = args_.get_one<Tensor>("--hidden_states", {ctx, inputs_});
            auto conditioning_embedding = args_.get_one<Tensor>("--conditioning_embedding", {ctx, inputs_});

            AdaLayerNormContinuous<> model(embedding_dim, conditioning_embedding_dim, elementwise_affine, eps, bias);

            CreateParametersVisitor create_parameters(ctx, inputs_, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            return model.forward(*ctx, hidden_states, conditioning_embedding);
        }

        if (args_.get(0) == "SpatialNorm") {
            auto f_channels = args_.get_one<int64_t>("--f_channels");
            auto zq_channels = args_.get_one<int64_t>("--zq_channels");
            auto f = args_.get_one<Tensor>("--f", {ctx, inputs_});
            auto zq = args_.get_one<Tensor>("--zq", {ctx, inputs_});

            SpatialNorm model(f_channels, zq_channels);

            CreateParametersVisitor create_parameters(ctx, inputs_, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            return model.forward(*ctx, f, zq);
        }

        if (args_.get(0) == "Upsample2D") {
            auto channels = args_.get_one<int64_t>("--channels");
            auto use_conv = args_.get_optional<bool>("--use_conv").value_or(false);
            auto out_channels = args_.get_optional<int64_t>("--out_channels");
            auto use_conv_transpose = args_.get_optional<bool>("--use_conv_transpose").value_or(false);
            auto hidden_states = args_.get_one<Tensor>("--hidden_states", {ctx, inputs_});

            Upsample2D model(channels, use_conv, out_channels, use_conv_transpose);

            CreateParametersVisitor create_parameters(ctx, inputs_, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            return model.forward(*ctx, hidden_states);
        }

        if (args_.get(0) == "Downsample2D") {
            auto channels = args_.get_one<int64_t>("--channels");
            auto use_conv = args_.get_optional<bool>("--use_conv").value_or(false);
            auto out_channels = args_.get_optional<int64_t>("--out_channels");
            auto padding = args_.get_optional<int64_t>("--padding").value_or(1);
            auto hidden_states = args_.get_one<Tensor>("--hidden_states", {ctx, inputs_});

            Downsample2D model(channels, use_conv, out_channels, padding);

            CreateParametersVisitor create_parameters(ctx, inputs_, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            return model.forward(*ctx, hidden_states);
        }

        if (args_.get(0) == "ResnetBlock2D") {
            auto in_channels = args_.get_one<int64_t>("--in_channels");
            auto out_channels = args_.get_optional<int64_t>("--out_channels");
            auto conv_shortcut = args_.get_optional<int64_t>("--conv_shortcut");
            auto dropout = args_.get_optional<float>("--dropout").value_or(0.0f);
            auto temb_channels = args_.get_optional<int64_t>("--temb_channels");
            auto groups = args_.get_optional<int64_t>("--groups").value_or(32);
            auto groups_out = args_.get_optional<int64_t>("--groups_out");
            auto eps = args_.get_optional<float>("--eps").value_or(1e-6f);
            auto non_linearity = args_.get_optional<std::string>("--non_linearity").value_or("swish");
            auto time_embedding_norm = args_.get_optional<std::string>("--time_embedding_norm").value_or("default");
            auto kernel = args_.get_optional<int64_t>("--kernel").value_or(3);
            auto output_scale_factor = args_.get_optional<int64_t>("--output_scale_factor").value_or(1);
            auto use_in_shortcut = args_.get_optional<bool>("--use_in_shortcut").value_or(false);
            auto up = args_.get_optional<bool>("--up").value_or(false);
            auto down = args_.get_optional<bool>("--down").value_or(false);
            auto conv_shortcut_bias = args_.get_optional<bool>("--conv_shortcut_bias").value_or(true);
            auto conv_2d_out_channels = args_.get_optional<int64_t>("--conv_2d_out_channels").value_or(0);
            auto hidden_states = args_.get_one<Tensor>("--hidden_states", {ctx, inputs_});
            auto temb = args_.get_optional<Tensor>("--temb", {ctx, inputs_});

            ResnetBlock2D model(
                in_channels,
                out_channels,
                conv_shortcut,
                dropout,
                temb_channels,
                groups,
                groups_out,
                eps,
                non_linearity,
                time_embedding_norm,
                kernel,
                output_scale_factor,
                use_in_shortcut,
                up,
                down,
                conv_shortcut_bias,
                conv_2d_out_channels
            );

            CreateParametersVisitor create_parameters(ctx, inputs_, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            return model.forward(*ctx, hidden_states, temb);
        }

        if (args_.get(0) == "TimestepEmbedding") {
            auto in_channels = args_.get_one<int64_t>("--in_channels");
            auto time_embed_dim = args_.get_one<int64_t>("--time_embed_dim");
            auto out_dim = args_.get_optional<int64_t>("--out_dim");
            auto cond_proj_dim = args_.get_optional<int64_t>("--cond_proj_dim");
            auto sample_proj_bias = args_.get_optional<bool>("--sample_proj_bias").value_or(true);
            auto sample = args_.get_one<Tensor>("--sample", {ctx, inputs_});
            auto condition = args_.get_optional<Tensor>("--condition", {ctx, inputs_});

            TimestepEmbedding<> model(in_channels, time_embed_dim, out_dim, cond_proj_dim, sample_proj_bias);

            CreateParametersVisitor create_parameters(ctx, inputs_, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            return model.forward(*ctx, sample, condition);
        }

        if (args_.get(0) == "Timesteps") {
            auto num_channels = args_.get_one<int64_t>("--num_channels");
            auto flip_sin_to_cos = args_.get_one<bool>("--flip_sin_to_cos");
            auto downscale_freq_shift = args_.get_one<float>("--downscale_freq_shift");
            auto scale = args_.get_optional<float>("--scale").value_or(1.0);
            auto timesteps = args_.get_one<Tensor>("--timesteps", {ctx, inputs_});

            Timesteps model(num_channels, flip_sin_to_cos, downscale_freq_shift, scale);

            CreateParametersVisitor create_parameters(ctx, inputs_, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            return model.forward(*ctx, timesteps);
        }

        if (args_.get(0) == "Attention") {
            auto query_dim = args_.get_one<int64_t>("--query_dim");
            auto heads = args_.get_one<int64_t>("--heads");
            auto dim_head = args_.get_one<int64_t>("--dim_head");
            auto dropout = args_.get_optional<float>("--dropout").value_or(0.0f);
            auto bias = args_.get_optional<bool>("--bias").value_or(false);
            auto residual_connection = args_.get_optional<bool>("--residual_connection").value_or(false);
            auto norm_num_groups = args_.get_optional<int64_t>("--norm_num_groups");
            auto eps = args_.get_optional<float>("--eps").value_or(1e-6);
            auto rescale_output_factor = args_.get_optional<float>("--rescale_output_factor").value_or(1.0f);
            auto upcast_softmax = args_.get_optional<bool>("--upcast_softmax").value_or(false);
            auto hidden_states = args_.get_one<Tensor>("--hidden_states", {ctx, inputs_});

            Attention<ScaledDotProductAttention<FlashAttentionOp>> model(
                query_dim,
                heads,
                dim_head,
                dropout,
                bias,
                residual_connection,
                norm_num_groups,
                eps,
                rescale_output_factor,
                upcast_softmax
            );

            CreateParametersVisitor create_parameters(ctx, inputs_, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            return model.forward(*ctx, hidden_states);
        }


        if (args_.get(0) == "UNetMidBlock2D") {
            auto in_channels = args_.get_one<int64_t>("--in_channels");
            auto resnet_eps = args_.get_optional<float>("--resnet_eps").value_or(1e-6);
            auto resnet_act_fn = args_.get_optional<std::string>("--resnet_act_fn").value_or("silu");
            auto output_scale_factor = args_.get_optional<float>("--output_scale_factor").value_or(1.0f);
            auto resnet_time_scale_shift = args_.get_optional<std::string>("--resnet_time_scale_shift").value_or("default");
            auto attention_head_dim = args_.get_optional<int64_t>("--attention_head_dim").value_or(1);
            auto resnet_groups = args_.get_optional<int64_t>("--resnet_groups").value_or(32);
            auto temb_channels = args_.get_optional<int64_t>("--temb_channels");
            auto add_attention = args_.get_optional<bool>("--add_attention").value_or(true);
            auto sample = args_.get_one<Tensor>("--sample", {ctx, inputs_});
            auto temb = args_.get_optional<Tensor>("--temb", {ctx, inputs_});

            UNetMidBlock2D model(
                in_channels,
                resnet_eps,
                resnet_act_fn,
                output_scale_factor,
                resnet_time_scale_shift,
                attention_head_dim,
                resnet_groups,
                temb_channels,
                add_attention
            );

            CreateParametersVisitor create_parameters(ctx, inputs_, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            return model.forward(*ctx, sample, temb);
        }


        if (args_.get(0) == "Flux2SwiGLU") {
            auto x = args_.get_one<Tensor>("--x", {ctx, inputs_});

            Flux2SwiGLU model;

            return model.forward(*ctx, x);
        }

        if (args_.get(0) == "Flux2FeedForward") {
            auto dim = args_.get_one<int64_t>("--dim");
            auto dim_out = args_.get_optional<int64_t>("--dim_out");
            auto mult = args_.get_optional<float>("--mult").value_or(3.0);
            auto inner_dim = args_.get_optional<int64_t>("--inner_dim");
            auto bias = args_.get_optional<bool>("--bias").value_or(false);
            auto x = args_.get_one<Tensor>("--x", {ctx, inputs_});

            Flux2FeedForward model(dim, dim_out, mult, inner_dim, bias);

            CreateParametersVisitor create_parameters(ctx, inputs_, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            return model.forward(*ctx, x);
        }

        if (args_.get(0) == "Flux2Modulation") {
            auto dim = args_.get_one<int64_t>("--dim");
            auto mod_param_sets = args_.get_optional<int64_t>("--mod_param_sets").value_or(2);
            auto bias = args_.get_optional<bool>("--bias").value_or(false);
            auto temb = args_.get_one<Tensor>("--temb", {ctx, inputs_});

            Flux2Modulation model(dim, mod_param_sets, bias);

            CreateParametersVisitor create_parameters(ctx, inputs_, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            return model.forward(*ctx, temb);
        }

        if (args_.get(0) == "Flux2TimestepGuidanceEmbeddings") {
            auto in_channels = args_.get_one<int64_t>("--in_channels");
            auto embedding_dim = args_.get_one<int64_t>("--embedding_dim");
            auto bias = args_.get_optional<bool>("--bias").value_or(false);
            auto guidance_embeds = args_.get_optional<bool>("--guidance_embeds").value_or(true);
            auto timestep = args_.get_one<Tensor>("--timestep", {ctx, inputs_});
            auto guidance = args_.get_optional<Tensor>("--guidance", {ctx, inputs_});

            Flux2TimestepGuidanceEmbeddings model(in_channels, embedding_dim, bias, guidance_embeds);

            CreateParametersVisitor create_parameters(ctx, inputs_, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            return model.forward(*ctx, timestep, guidance);
        }

        if (args_.get(0) == "Flux2PosEmbed") {
            auto theta = args_.get_one<int64_t>("--theta");
            auto axes_dim = args_.get_many<int64_t>("--axes_dim");
            auto x = args_.get_one<Tensor>("--x", {ctx, inputs_});
            auto position_ids = args_.get_one<Tensor>("--position_ids", {ctx, inputs_});

            Flux2PosEmbed model(theta, axes_dim);

            CreateParametersVisitor create_parameters(ctx, inputs_, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            return model.forward(*ctx, x, position_ids);
        }

        if (args_.get(0) == "Flux2Attention") {
            auto query_dim = args_.get_one<int64_t>("--query_dim");
            auto heads = args_.get_optional<int64_t>("--heads").value_or(8);
            auto dim_head = args_.get_optional<int64_t>("--dim_head").value_or(64);
            auto dropout = args_.get_optional<float>("--dropout").value_or(0.0);
            auto bias = args_.get_optional<bool>("--bias").value_or(false);
            auto added_kv_proj_dim = args_.get_optional<int64_t>("--added_kv_proj_dim");
            auto added_proj_bias = args_.get_optional<bool>("--added_proj_bias").value_or(true);
            auto out_bias = args_.get_optional<bool>("--out_bias").value_or(true);
            auto eps = args_.get_optional<float>("--eps").value_or(1e-5);
            auto out_dim = args_.get_optional<int64_t>("--out_dim");
            auto elementwise_affine = args_.get_optional<bool>("--elementwise_affine").value_or(true);
            auto hidden_states = args_.get_one<Tensor>("--hidden_states", {ctx, inputs_});
            auto encoder_hidden_states = args_.get_optional<Tensor>("--encoder_hidden_states", {ctx, inputs_});
            auto attention_mask = args_.get_optional<Tensor>("--attention_mask", {ctx, inputs_});
            auto theta = args_.get_optional<int64_t>("--image_rotary_emb-theta");
            auto axes_dim = args_.get_many<int64_t>("--image_rotary_emb-axes_dim");
            auto position_ids = args_.get_optional<Tensor>("--image_rotary_emb-position_ids", {ctx, inputs_});

            auto image_rotary_emb = theta && position_ids && !axes_dim.empty() ? std::make_optional(std::make_pair(
                std::make_shared<Flux2PosEmbed>(*theta, axes_dim),
                *position_ids
            )) : std::nullopt;

            Flux2Attention<ScaledDotProductAttention<FlashAttentionOp>> model(
                query_dim,
                heads,
                dim_head,
                dropout,
                bias,
                added_kv_proj_dim,
                added_proj_bias,
                out_bias,
                eps,
                out_dim,
                elementwise_affine
            );

            CreateParametersVisitor create_parameters(ctx, inputs_, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto [y1, y2] = model.forward(*ctx, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb);
            std::vector<Tensor> results;

            results.push_back(y1);

            if (y2)
                results.push_back(*y2);

            return results;
        }

        if (args_.get(0) == "Flux2ParallelSelfAttention") {
            auto query_dim = args_.get_one<int64_t>("--query_dim");
            auto heads = args_.get_optional<int64_t>("--heads").value_or(8);
            auto dim_head = args_.get_optional<int64_t>("--dim_head").value_or(64);
            auto dropout = args_.get_optional<float>("--dropout").value_or(0.0);
            auto bias = args_.get_optional<bool>("--bias").value_or(false);
            auto out_bias = args_.get_optional<bool>("--out_bias").value_or(true);
            auto eps = args_.get_optional<float>("--eps").value_or(1e-5);
            auto out_dim = args_.get_optional<int64_t>("--out_dim");
            auto elementwise_affine = args_.get_optional<bool>("--elementwise_affine").value_or(true);
            auto mlp_ratio = args_.get_optional<float>("--mlp_ratio").value_or(4.0);
            auto mlp_mult_factor = args_.get_optional<int64_t>("--mlp_mult_factor").value_or(2);
            auto hidden_states = args_.get_one<Tensor>("--hidden_states", {ctx, inputs_});
            auto attention_mask = args_.get_optional<Tensor>("--attention_mask", {ctx, inputs_});
            auto theta = args_.get_optional<int64_t>("--image_rotary_emb-theta");
            auto axes_dim = args_.get_many<int64_t>("--image_rotary_emb-axes_dim");
            auto position_ids = args_.get_optional<Tensor>("--image_rotary_emb-position_ids", {ctx, inputs_});

            auto image_rotary_emb = theta && position_ids && !axes_dim.empty() ? std::make_optional(std::make_pair(
                std::make_shared<Flux2PosEmbed>(*theta, axes_dim),
                *position_ids
            )) : std::nullopt;

            Flux2ParallelSelfAttention<ScaledDotProductAttention<FlashAttentionOp>> model(
                query_dim,
                heads,
                dim_head,
                dropout,
                bias,
                out_bias,
                eps,
                out_dim,
                elementwise_affine,
                mlp_ratio,
                mlp_mult_factor
            );

            CreateParametersVisitor create_parameters(ctx, inputs_, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            return model.forward(*ctx, hidden_states, attention_mask, image_rotary_emb);
        }

        if (args_.get(0) == "Flux2SingleTransformerBlock") {
            auto dim = args_.get_one<int64_t>("--dim");
            auto num_attention_heads = args_.get_one<int64_t>("--num_attention_heads");
            auto attention_head_dim = args_.get_one<int64_t>("--attention_head_dim");
            auto mlp_ratio = args_.get_optional<float>("--mlp_ratio").value_or(3.0);
            auto eps = args_.get_optional<float>("--eps").value_or(1e-6);
            auto bias = args_.get_optional<bool>("--bias").value_or(false);
            auto hidden_states = args_.get_one<Tensor>("--hidden_states", {ctx, inputs_});
            auto encoder_hidden_states = args_.get_optional<Tensor>("--encoder_hidden_states", {ctx, inputs_});
            auto temb_mod = args_.get_one<Tensor>("--temb_mod", {ctx, inputs_});
            auto split_hidden_states = args_.get_optional<bool>("--split_hidden_states").value_or(false);
            auto text_seq_len = args_.get_optional<int64_t>("--text_seq_len");
            auto theta = args_.get_optional<int64_t>("--image_rotary_emb-theta");
            auto axes_dim = args_.get_many<int64_t>("--image_rotary_emb-axes_dim");
            auto position_ids = args_.get_optional<Tensor>("--image_rotary_emb-position_ids", {ctx, inputs_});

            auto image_rotary_emb = theta && position_ids && !axes_dim.empty() ? std::make_optional(std::make_pair(
                std::make_shared<Flux2PosEmbed>(*theta, axes_dim),
                *position_ids
            )) : std::nullopt;

            Flux2SingleTransformerBlock<ScaledDotProductAttention<FlashAttentionOp>> model(
                dim,
                num_attention_heads,
                attention_head_dim,
                mlp_ratio,
                eps,
                bias
            );

            CreateParametersVisitor create_parameters(ctx, inputs_, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto [y1, y2] = model.forward(
                *ctx,
                hidden_states,
                encoder_hidden_states,
                temb_mod,
                image_rotary_emb,
                split_hidden_states,
                text_seq_len
            );

            std::vector<Tensor> results;

            results.push_back(y1);

            if (y2)
                results.push_back(*y2);

            return results;
        }

        if (args_.get(0) == "Flux2TransformerBlock") {
            auto dim = args_.get_one<int64_t>("--dim");
            auto num_attention_heads = args_.get_one<int64_t>("--num_attention_heads");
            auto attention_head_dim = args_.get_one<int64_t>("--attention_head_dim");
            auto mlp_ratio = args_.get_optional<float>("--mlp_ratio").value_or(3.0);
            auto eps = args_.get_optional<float>("--eps").value_or(1e-6);
            auto bias = args_.get_optional<bool>("--bias").value_or(false);
            auto hidden_states = args_.get_one<Tensor>("--hidden_states", {ctx, inputs_});
            auto encoder_hidden_states = args_.get_one<Tensor>("--encoder_hidden_states", {ctx, inputs_});
            auto temb_mod_img = args_.get_one<Tensor>("--temb_mod_img", {ctx, inputs_});
            auto temb_mod_txt = args_.get_one<Tensor>("--temb_mod_txt", {ctx, inputs_});
            auto theta = args_.get_optional<int64_t>("--image_rotary_emb-theta");
            auto axes_dim = args_.get_many<int64_t>("--image_rotary_emb-axes_dim");
            auto position_ids = args_.get_optional<Tensor>("--image_rotary_emb-position_ids", {ctx, inputs_});

            auto image_rotary_emb = theta && position_ids && !axes_dim.empty() ? std::make_optional(std::make_pair(
                std::make_shared<Flux2PosEmbed>(*theta, axes_dim),
                *position_ids
            )) : std::nullopt;

            Flux2TransformerBlock<ScaledDotProductAttention<FlashAttentionOp>> model(
                dim,
                num_attention_heads,
                attention_head_dim,
                mlp_ratio,
                eps,
                bias
            );

            CreateParametersVisitor create_parameters(ctx, inputs_, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto [y1, y2] = model.forward(
                *ctx,
                hidden_states,
                encoder_hidden_states,
                temb_mod_img,
                temb_mod_txt,
                image_rotary_emb
            );

            return {{y1, y2}};
        }

        if (args_.get(0) == "Flux2Transformer2DModel") {
            auto patch_size = args_.get_optional<int64_t>("--patch_size").value_or(1);
            auto in_channels = args_.get_optional<int64_t>("--in_channels").value_or(128);
            auto out_channels = args_.get_optional<int64_t>("--out_channels");
            auto num_layers = args_.get_optional<int64_t>("--num_layers").value_or(8);
            auto num_single_layers = args_.get_optional<int64_t>("--num_single_layers").value_or(48);
            auto attention_head_dim = args_.get_optional<int64_t>("--attention_head_dim").value_or(128);
            auto num_attention_heads = args_.get_optional<int64_t>("--num_attention_heads").value_or(48);
            auto joint_attention_dim = args_.get_optional<int64_t>("--joint_attention_dim").value_or(15360);
            auto timestep_guidance_channels = args_.get_optional<int64_t>("--timestep_guidance_channels").value_or(256);
            auto mlp_ratio = args_.get_optional<float>("--mlp_ratio").value_or(3.0f);
            auto axes_dims_rope = args_.get_many<int64_t>("--axes_dims_rope");
            auto rope_theta = args_.get_optional<int64_t>("--rope_theta").value_or(2000);
            auto eps = args_.get_optional<float>("--eps").value_or(1e-6f);
            auto guidance_embeds = args_.get_optional<bool>("--guidance_embeds").value_or(true);

            auto hidden_states = args_.get_one<Tensor>("--hidden_states", {ctx, inputs_});
            auto encoder_hidden_states = args_.get_one<Tensor>("--encoder_hidden_states", {ctx, inputs_});
            auto timestep = args_.get_one<Tensor>("--timestep", {ctx, inputs_});
            auto img_ids = args_.get_one<Tensor>("--img_ids", {ctx, inputs_});
            auto txt_ids = args_.get_one<Tensor>("--txt_ids", {ctx, inputs_});
            auto guidance = args_.get_optional<Tensor>("--guidance", {ctx, inputs_});
            auto num_ref_tokens = args_.get_optional<int64_t>("--num_ref_tokens").value_or(0);
            auto ref_fixed_timestep = args_.get_optional<float>("--ref_fixed_timestep").value_or(0.0f);

            if (axes_dims_rope.empty())
                axes_dims_rope = std::vector<int64_t>{32, 32, 32, 32};

            Flux2Transformer2DModel model(
                patch_size,
                in_channels,
                out_channels,
                num_layers,
                num_single_layers,
                attention_head_dim,
                num_attention_heads,
                joint_attention_dim,
                timestep_guidance_channels,
                mlp_ratio,
                axes_dims_rope,
                rope_theta,
                eps,
                guidance_embeds
            );

            CreateParametersVisitor create_parameters(ctx, inputs_, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            return model.forward(
                *ctx,
                hidden_states,
                encoder_hidden_states,
                timestep,
                img_ids,
                txt_ids,
                guidance,
                //std::nullopt,   // kv_cache
                //std::nullopt,   // kv_cache_mode
                num_ref_tokens,
                ref_fixed_timestep
            );
        }

        throw std::runtime_error("Uknown command: " + args_.get(0));
    }

protected:
    class CreateParametersVisitor : public Visitor {
    public:
        CreateParametersVisitor(Context& ctx, std::vector<std::pair<Tensor, std::vector<float>>>& inputs, ArgumentParser& args)
            : ctx_(ctx), inputs_(inputs), args_(args)
        {}

        virtual void visit(Parameter& parameter, std::vector<std::string> path) {
            auto joined_path = join_path(path);
            auto tensor = args_.get_one<Tensor>(joined_path, {ctx_, inputs_});
            parameter.set(tensor);
        }

    private:
        Context& ctx_;
        std::vector<std::pair<Tensor, std::vector<float>>>& inputs_;
        ArgumentParser& args_;
        
        static std::string join_path(const std::vector<std::string>& path) {
            return std::accumulate(std::begin(path), std::end(path), std::string("--param"), [](const std::string& acc, const std::string& x) {
                return acc + "-" + x;
            });
        }
    };
};

int main(int argc, char** argv) {
    TestDiffusersCLI cli(argc, argv);
    return cli.main();
}
