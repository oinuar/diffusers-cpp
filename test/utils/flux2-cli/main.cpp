#include "../TestCLI.hpp"
#include "nn/RethrowVisitor.hpp"
#include "nn/attention/ScaledDotProductAttention.hpp"
#include "nn/attention/FlashAttentionOp.hpp"

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

#include "transformers/models/qwen3/Qwen3Config.hpp"
#include "diffusers/pipelines/flux2/Flux2KleinPipeline.hpp"

#include <numeric>

class TestFlux2CLI : public TestCLI {
public:
    TestFlux2CLI(int argc, char** argv) : TestCLI(argc, argv) {}

    virtual std::vector<Tensor> compute(Runtime& runtime) {

        if (args_.get(0) == "Flux2SwiGLU") {
            auto x = args_.get_one<Tensor>("--x", {runtime});

            Flux2SwiGLU model;

            auto output = model.forward(runtime, x);

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "Flux2FeedForward") {
            auto dim = args_.get_one<int64_t>("--dim");
            auto dim_out = args_.get_optional<int64_t>("--dim_out");
            auto mult = args_.get_optional<float>("--mult").value_or(3.0);
            auto inner_dim = args_.get_optional<int64_t>("--inner_dim");
            auto bias = args_.get_optional<bool>("--bias").value_or(false);
            auto x = args_.get_one<Tensor>("--x", {runtime});

            Flux2FeedForward model(dim, dim_out, mult, inner_dim, bias);

            CreateParametersVisitor create_parameters(runtime, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto output = model.forward(runtime, x);

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "Flux2Modulation") {
            auto dim = args_.get_one<int64_t>("--dim");
            auto mod_param_sets = args_.get_optional<int64_t>("--mod_param_sets").value_or(2);
            auto bias = args_.get_optional<bool>("--bias").value_or(false);
            auto temb = args_.get_one<Tensor>("--temb", {runtime});

            Flux2Modulation model(dim, mod_param_sets, bias);

            CreateParametersVisitor create_parameters(runtime, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto output = model.forward(runtime, temb);

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "Flux2TimestepGuidanceEmbeddings") {
            auto in_channels = args_.get_one<int64_t>("--in_channels");
            auto embedding_dim = args_.get_one<int64_t>("--embedding_dim");
            auto bias = args_.get_optional<bool>("--bias").value_or(false);
            auto guidance_embeds = args_.get_optional<bool>("--guidance_embeds").value_or(true);
            auto timestep = args_.get_one<Tensor>("--timestep", {runtime});
            auto guidance = args_.get_optional<Tensor>("--guidance", {runtime});

            Flux2TimestepGuidanceEmbeddings model(in_channels, embedding_dim, bias, guidance_embeds);

            CreateParametersVisitor create_parameters(runtime, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto output = model.forward(runtime, timestep, guidance);

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "Flux2PosEmbed") {
            auto theta = args_.get_one<int64_t>("--theta");
            auto axes_dim = args_.get_many<int64_t>("--axes_dim");
            auto x = args_.get_one<Tensor>("--x", {runtime});
            auto position_ids = args_.get_one<Tensor>("--position_ids", {runtime});

            Flux2PosEmbed model(theta, axes_dim);

            CreateParametersVisitor create_parameters(runtime, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto output = model.forward(runtime, x, position_ids);

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
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
            auto hidden_states = args_.get_one<Tensor>("--hidden_states", {runtime});
            auto encoder_hidden_states = args_.get_optional<Tensor>("--encoder_hidden_states", {runtime});
            auto attention_mask = args_.get_optional<Tensor>("--attention_mask", {runtime});
            auto theta = args_.get_optional<int64_t>("--image_rotary_emb-theta");
            auto axes_dim = args_.get_many<int64_t>("--image_rotary_emb-axes_dim");
            auto position_ids = args_.get_optional<Tensor>("--image_rotary_emb-position_ids", {runtime});

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

            CreateParametersVisitor create_parameters(runtime, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto [y1, y2] = model.forward(runtime, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb);
            std::vector<Tensor> results;

            results.push_back(y1);

            if (y2)
                results.push_back(*y2);

            Graph graph(runtime, std::move(results));
            Computation computation(graph);
            return computation.results();
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
            auto hidden_states = args_.get_one<Tensor>("--hidden_states", {runtime});
            auto attention_mask = args_.get_optional<Tensor>("--attention_mask", {runtime});
            auto theta = args_.get_optional<int64_t>("--image_rotary_emb-theta");
            auto axes_dim = args_.get_many<int64_t>("--image_rotary_emb-axes_dim");
            auto position_ids = args_.get_optional<Tensor>("--image_rotary_emb-position_ids", {runtime});

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

            CreateParametersVisitor create_parameters(runtime, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto output = model.forward(runtime, hidden_states, attention_mask, image_rotary_emb);

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "Flux2SingleTransformerBlock") {
            auto dim = args_.get_one<int64_t>("--dim");
            auto num_attention_heads = args_.get_one<int64_t>("--num_attention_heads");
            auto attention_head_dim = args_.get_one<int64_t>("--attention_head_dim");
            auto mlp_ratio = args_.get_optional<float>("--mlp_ratio").value_or(3.0);
            auto eps = args_.get_optional<float>("--eps").value_or(1e-6);
            auto bias = args_.get_optional<bool>("--bias").value_or(false);
            auto hidden_states = args_.get_one<Tensor>("--hidden_states", {runtime});
            auto encoder_hidden_states = args_.get_optional<Tensor>("--encoder_hidden_states", {runtime});
            auto temb_mod = args_.get_one<Tensor>("--temb_mod", {runtime});
            auto split_hidden_states = args_.get_optional<bool>("--split_hidden_states").value_or(false);
            auto text_seq_len = args_.get_optional<int64_t>("--text_seq_len");
            auto theta = args_.get_optional<int64_t>("--image_rotary_emb-theta");
            auto axes_dim = args_.get_many<int64_t>("--image_rotary_emb-axes_dim");
            auto position_ids = args_.get_optional<Tensor>("--image_rotary_emb-position_ids", {runtime});

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

            CreateParametersVisitor create_parameters(runtime, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto [y1, y2] = model.forward(
                runtime,
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

            Graph graph(runtime, std::move(results));
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "Flux2TransformerBlock") {
            auto dim = args_.get_one<int64_t>("--dim");
            auto num_attention_heads = args_.get_one<int64_t>("--num_attention_heads");
            auto attention_head_dim = args_.get_one<int64_t>("--attention_head_dim");
            auto mlp_ratio = args_.get_optional<float>("--mlp_ratio").value_or(3.0);
            auto eps = args_.get_optional<float>("--eps").value_or(1e-6);
            auto bias = args_.get_optional<bool>("--bias").value_or(false);
            auto hidden_states = args_.get_one<Tensor>("--hidden_states", {runtime});
            auto encoder_hidden_states = args_.get_one<Tensor>("--encoder_hidden_states", {runtime});
            auto temb_mod_img = args_.get_one<Tensor>("--temb_mod_img", {runtime});
            auto temb_mod_txt = args_.get_one<Tensor>("--temb_mod_txt", {runtime});
            auto theta = args_.get_optional<int64_t>("--image_rotary_emb-theta");
            auto axes_dim = args_.get_many<int64_t>("--image_rotary_emb-axes_dim");
            auto position_ids = args_.get_optional<Tensor>("--image_rotary_emb-position_ids", {runtime});

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

            CreateParametersVisitor create_parameters(runtime, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto [y1, y2] = model.forward(
                runtime,
                hidden_states,
                encoder_hidden_states,
                temb_mod_img,
                temb_mod_txt,
                image_rotary_emb
            );

            Graph graph(runtime, {y1, y2});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "Flux2Transformer2DModel") {
            Flux2Transformer2DModel::Config config;

            config.patch_size = args_.get_optional<int64_t>("--patch_size").value_or(config.patch_size);
            config.in_channels = args_.get_optional<int64_t>("--in_channels").value_or(config.in_channels);
            config.out_channels = args_.get_optional<int64_t>("--out_channels");
            config.num_layers = args_.get_optional<int64_t>("--num_layers").value_or(config.num_layers);
            config.num_single_layers = args_.get_optional<int64_t>("--num_single_layers").value_or(config.num_single_layers);
            config.attention_head_dim = args_.get_optional<int64_t>("--attention_head_dim").value_or(config.attention_head_dim);
            config.num_attention_heads = args_.get_optional<int64_t>("--num_attention_heads").value_or(config.num_attention_heads);
            config.joint_attention_dim = args_.get_optional<int64_t>("--joint_attention_dim").value_or(config.joint_attention_dim);
            config.timestep_guidance_channels = args_.get_optional<int64_t>("--timestep_guidance_channels").value_or(config.timestep_guidance_channels);
            config.mlp_ratio = args_.get_optional<float>("--mlp_ratio").value_or(config.mlp_ratio);
            auto axes_dims_rope = args_.get_many<int64_t>("--axes_dims_rope");
            config.rope_theta = args_.get_optional<int64_t>("--rope_theta").value_or(config.rope_theta);
            config.eps = args_.get_optional<float>("--eps").value_or(config.eps);
            config.guidance_embeds = args_.get_optional<bool>("--guidance_embeds").value_or(config.guidance_embeds);

            auto hidden_states = args_.get_one<Tensor>("--hidden_states", {runtime});
            auto encoder_hidden_states = args_.get_one<Tensor>("--encoder_hidden_states", {runtime});
            auto timestep = args_.get_one<Tensor>("--timestep", {runtime});
            auto img_ids = args_.get_one<Tensor>("--img_ids", {runtime});
            auto txt_ids = args_.get_one<Tensor>("--txt_ids", {runtime});
            auto guidance = args_.get_optional<Tensor>("--guidance", {runtime});
            auto num_ref_tokens = args_.get_optional<int64_t>("--num_ref_tokens").value_or(0);
            auto ref_fixed_timestep = args_.get_optional<float>("--ref_fixed_timestep").value_or(0.0f);

            if (!axes_dims_rope.empty())
                config.axes_dims_rope = axes_dims_rope;

            Flux2Transformer2DModel model(config);

            CreateParametersVisitor create_parameters(runtime, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto output = model.forward(
                runtime,
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

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "Flux2KleinPipeline_pack_latents" ||
            args_.get(0) == "Flux2KleinPipeline_unpack_latents" ||
            args_.get(0) == "Flux2KleinPipeline_patchify_latents" ||
            args_.get(0) == "Flux2KleinPipeline_unpatchify_latents") {
            auto [latents_shape, latents_data] = ArgumentParser::parser<Tensor>::TensorParser(args_.get_one<std::string>("--latents")).parse();
            auto latents = runtime.create<float>(latents_shape, [latents_data](std::mt19937&) { return std::move(latents_data); });

            Tensor result;

            if (args_.get(0) == "Flux2KleinPipeline_pack_latents") {
                result = Flux2KleinPipeline::pack_latents(latents);
            } else if (args_.get(0) == "Flux2KleinPipeline_unpack_latents") {
                result = Flux2KleinPipeline::unpack_latents(
                    latents,
                    args_.get_one<int>("--packed_h"),
                    args_.get_one<int>("--packed_w")
                );
            } else if (args_.get(0) == "Flux2KleinPipeline_patchify_latents") {
                result = Flux2KleinPipeline::patchify_latents(
                    latents,
                    args_.get_one<int>("--channels"),
                    args_.get_one<int>("--packed_h"),
                    args_.get_one<int>("--packed_w")
                );
            } else {
                result = Flux2KleinPipeline::unpatchify_latents(
                    latents,
                    args_.get_one<int>("--channels"),
                    args_.get_one<int>("--packed_h"),
                    args_.get_one<int>("--packed_w")
                );
            }

            Graph graph(runtime, {result});
            Computation computation(graph);

            return computation.results();
        }

        if (args_.get(0).rfind("Flux2KleinPipeline", 0) == 0) {
            Flux2Transformer2DModel::Config transformer_config;
            {
                transformer_config.patch_size = args_.get_optional<int64_t>("--transformer-patch_size").value_or(transformer_config.patch_size);
                transformer_config.in_channels = args_.get_optional<int64_t>("--transformer-in_channels").value_or(transformer_config.in_channels);
                transformer_config.out_channels = args_.get_optional<int64_t>("--transformer-out_channels");
                transformer_config.num_layers = args_.get_optional<int64_t>("--transformer-num_layers").value_or(transformer_config.num_layers);
                transformer_config.num_single_layers = args_.get_optional<int64_t>("--transformer-num_single_layers").value_or(transformer_config.num_single_layers);
                transformer_config.attention_head_dim = args_.get_optional<int64_t>("--transformer-attention_head_dim").value_or(transformer_config.attention_head_dim);
                transformer_config.num_attention_heads = args_.get_optional<int64_t>("--transformer-num_attention_heads").value_or(transformer_config.num_attention_heads);
                transformer_config.joint_attention_dim = args_.get_optional<int64_t>("--transformer-joint_attention_dim").value_or(transformer_config.joint_attention_dim);
                transformer_config.timestep_guidance_channels = args_.get_optional<int64_t>("--transformer-timestep_guidance_channels").value_or(transformer_config.timestep_guidance_channels);
                transformer_config.mlp_ratio = args_.get_optional<float>("--transformer-mlp_ratio").value_or(transformer_config.mlp_ratio);
                auto axes_dims_rope = args_.get_many<int64_t>("--transformer-axes_dims_rope");
                transformer_config.rope_theta = args_.get_optional<int64_t>("--transformer-rope_theta").value_or(transformer_config.rope_theta);
                transformer_config.eps = args_.get_optional<float>("--transformer-eps").value_or(transformer_config.eps);
                transformer_config.guidance_embeds = args_.get_optional<bool>("--transformer-guidance_embeds").value_or(transformer_config.guidance_embeds);

                if (!axes_dims_rope.empty())
                    transformer_config.axes_dims_rope = axes_dims_rope;
            }

            AutoencoderKLFlux2::Config vae_config;
            {
                vae_config.in_channels = args_.get_optional<int64_t>("--vae-in_channels").value_or(vae_config.in_channels);
                vae_config.out_channels = args_.get_optional<int64_t>("--vae-out_channels").value_or(vae_config.out_channels);
                auto block_out_channels = args_.get_many<int64_t>("--vae-block_out_channels");
                vae_config.layers_per_block = args_.get_optional<int64_t>("--vae-layers_per_block").value_or(vae_config.layers_per_block);
                vae_config.latent_channels = args_.get_optional<int64_t>("--vae-latent_channels").value_or(vae_config.latent_channels);
                vae_config.norm_num_groups = args_.get_optional<int64_t>("--vae-norm_num_groups").value_or(vae_config.norm_num_groups);
                vae_config.sample_size = args_.get_optional<int64_t>("--vae-sample_size").value_or(vae_config.sample_size);
                vae_config.force_upcast = args_.get_optional<bool>("--vae-force_upcast").value_or(vae_config.force_upcast);
                vae_config.use_quant_conv = args_.get_optional<bool>("--vae-use_quant_conv").value_or(vae_config.use_quant_conv);
                vae_config.use_post_quant_conv = args_.get_optional<bool>("--vae-use_post_quant_conv").value_or(vae_config.use_post_quant_conv);
                vae_config.mid_block_add_attention = args_.get_optional<bool>("--vae-mid_block_add_attention").value_or(vae_config.mid_block_add_attention);
                vae_config.batch_norm_eps = args_.get_optional<float>("--vae-batch_norm_eps").value_or(vae_config.batch_norm_eps);
                vae_config.batch_norm_momentum = args_.get_optional<float>("--vae-batch_norm_momentum").value_or(vae_config.batch_norm_momentum);
                vae_config.patch_size = std::make_tuple(
                    args_.get_optional<int64_t>("--vae-patch_size-0").value_or(std::get<0>(vae_config.patch_size)),
                    args_.get_optional<int64_t>("--vae-patch_size-1").value_or(std::get<1>(vae_config.patch_size))
                );

                if (!block_out_channels.empty())
                    vae_config.block_out_channels = block_out_channels;
            }

            Qwen3Config qwen_config;
            {
                qwen_config.vocab_size = args_.get_optional<int64_t>("--text_encoder-vocab_size").value_or(qwen_config.vocab_size);
                qwen_config.hidden_size = args_.get_optional<int64_t>("--text_encoder-hidden_size").value_or(qwen_config.hidden_size);
                qwen_config.intermediate_size = args_.get_optional<int64_t>("--text_encoder-intermediate_size").value_or(qwen_config.intermediate_size);
                qwen_config.num_hidden_layers = args_.get_optional<int64_t>("--text_encoder-num_hidden_layers").value_or(qwen_config.num_hidden_layers);
                qwen_config.num_attention_heads = args_.get_optional<int64_t>("--text_encoder-num_attention_heads").value_or(qwen_config.num_attention_heads);
                qwen_config.num_key_value_heads = args_.get_optional<int64_t>("--text_encoder-num_key_value_heads").value_or(qwen_config.num_key_value_heads);
                qwen_config.max_position_embeddings = args_.get_optional<int64_t>("--text_encoder-max_position_embeddings").value_or(qwen_config.max_position_embeddings);
            }

            auto tokenizer_dir = args_.get_one<std::string>("--tokenizer_dir");

            Flux2Transformer2DModel transformer(transformer_config);
            {
                CreateParametersVisitor create_parameters(runtime, args_, "transformer");
                RethrowVisitor visitor(create_parameters);
                transformer.accept(visitor);
                visitor.rethrow();
            }
            
            AutoencoderKLFlux2 vae(vae_config);
            {
                CreateParametersVisitor create_parameters(runtime, args_, "vae");
                RethrowVisitor visitor(create_parameters);
                vae.accept(visitor);
                visitor.rethrow();
            }

            Qwen3ForCausalLM text_encoder(qwen_config);
            {
                CreateParametersVisitor create_parameters(runtime, args_, "text_encoder");
                RethrowVisitor visitor(create_parameters);
                text_encoder.accept(visitor);
                visitor.rethrow();
            }

            auto tokenizer = Qwen2TokenizerFast::from_pretrained(tokenizer_dir);

            Flux2KleinPipeline pipeline(
                std::move(transformer),
                std::move(vae),
                std::move(text_encoder),
                std::move(tokenizer)
            );

            if (args_.get(0) == "Flux2KleinPipeline_embeddings") {
                auto batch = args_.get_one<int>("--batch");
                auto prompt = args_.get_one<std::string>("--prompt");
                auto max_sequence_length = args_.get_one<int>("--max_sequence_length");
                auto packed_h = args_.get_one<int>("--packed_h");
                auto packed_w = args_.get_one<int>("--packed_w");
                auto images = args_.get_many<Image>("--images");

                auto [
                    graph,
                    prompt_embeds,
                    txt_ids,
                    img_ids,
                    image_latents_concat,
                    image_latent_ids_concat
                ] = std::move(pipeline.make_embeddings_graph(
                    runtime,
                    prompt,
                    max_sequence_length,
                    batch,
                    packed_h,
                    packed_w,
                    images
                ));

                Computation computation(graph);
                
                std::vector<Tensor> results = {
                    prompt_embeds,
                    txt_ids,
                    img_ids
                };

                if (image_latents_concat)
                    results.push_back(*image_latents_concat);
                
                if (image_latent_ids_concat)
                    results.push_back(*image_latent_ids_concat);

                return std::move(results);
            }

            if (args_.get(0) == "Flux2KleinPipeline_denoise") {
                auto batch = args_.get_one<int>("--batch");
                auto packed_h = args_.get_one<int>("--packed_h");
                auto packed_w = args_.get_one<int>("--packed_w");
                auto num_ref_tokens = args_.get_one<int>("--num_ref_tokens");
                auto timestep = args_.get_one<float>("--timestep");
                auto dt = args_.get_one<float>("--dt");
                auto [_, init_latents] = ArgumentParser::parser<Tensor>::TensorParser(args_.get_one<std::string>("--init_latents")).parse();
                auto [prompt_embeds_shape, prompt_embeds_data] = ArgumentParser::parser<Tensor>::TensorParser(args_.get_one<std::string>("--prompt_embeds")).parse();
                auto [img_ids_shape, img_ids_data] = ArgumentParser::parser<Tensor>::TensorParser(args_.get_one<std::string>("--img_ids")).parse();
                auto [txt_ids_shape, txt_ids_data] = ArgumentParser::parser<Tensor>::TensorParser(args_.get_one<std::string>("--txt_ids")).parse();

                Tensor::Shape image_latents_shape;
                Tensor::Shape image_latent_ids_shape;
                std::vector<float> image_latents_data;
                std::vector<float> image_latent_ids_data;

                if (num_ref_tokens > 0) {
                    auto image_latents = ArgumentParser::parser<Tensor>::TensorParser(args_.get_one<std::string>("--image_latents")).parse();
                    auto image_latent_ids = ArgumentParser::parser<Tensor>::TensorParser(args_.get_one<std::string>("--image_latent_ids")).parse();

                    image_latents_shape = image_latents.first;
                    image_latent_ids_shape = image_latent_ids.first;
                    image_latents_data = std::move(image_latents.second);
                    image_latent_ids_data = std::move(image_latent_ids.second);
                }

                auto [graph, latents, next_latents] = std::move(pipeline.make_denoise_graph(
                    runtime,
                    batch,
                    packed_h,
                    packed_w,
                    num_ref_tokens,
                    prompt_embeds_shape,
                    img_ids_shape,
                    txt_ids_shape,
                    image_latents_shape,
                    image_latent_ids_shape,
                    &init_latents,
                    &prompt_embeds_data,
                    &img_ids_data,
                    &txt_ids_data,
                    &image_latents_data,
                    &image_latent_ids_data,
                    &timestep,
                    &dt
                ));

                Computation computation(graph);

                return {next_latents};
            }

            if (args_.get(0) == "Flux2KleinPipeline_decode") {
                auto packed_h = args_.get_one<int>("--packed_h");
                auto packed_w = args_.get_one<int>("--packed_w");
                auto [latents_shape, latents_data] = ArgumentParser::parser<Tensor>::TensorParser(args_.get_one<std::string>("--latents")).parse();

                auto graph = std::move(pipeline.make_decode_graph(
                    runtime,
                    packed_h,
                    packed_w,
                    latents_shape,
                    &latents_data
                ));

                Computation computation(graph);
                
                return computation.results();
            }

            /*if (args_.get(0) == "Flux2KleinPipeline_call") {
                Flux2KleinPipeline::GenerationOptions options;

                options.prompt = args_.get_one<std::string>("--prompt");
                options.height = args_.get_one<int>("--height");
                options.width = args_.get_one<int>("--width");
                options.num_inference_steps = args_.get_one<int>("--num_inference_steps");
                options.max_sequence_length = args_.get_one<int>("--max_sequence_length");

                if (auto init_latents = args_.get_optional<std::string>("--init_latents"))
                    options.init_latents = std::move(
                        ArgumentParser::parser<Tensor>::TensorParser(*init_latents).parse().second);

                auto images = pipeline(runtime.scheduler(), std::move(options));
                std::vector<Tensor> results;

                for (const auto& image : images) {
                    std::vector<float> pixels(image.pixels().begin(), image.pixels().end());

                    auto tensor = runtime.create<float>({(int64_t)image.height(), (int64_t)image.width(), (int64_t)image.channels()}, [pixels](std::mt19937&) {
                        return std::move(pixels);
                    });

                    results.push_back(tensor);
                }

                Graph graph(runtime, std::move(results));
                Computation computation(graph);

                return computation.results();
            }*/
        }

        throw std::runtime_error("Uknown command: " + args_.get(0));
    }

    virtual size_t get_graph_size() const {
        if (args_.get(0) == "Flux2KleinPipeline_embeddings" ||
            args_.get(0) == "Flux2KleinPipeline_call")
            return 65536;
        
        return TestCLI::get_graph_size();
    }
};

int main(int argc, char** argv) {
    TestFlux2CLI cli(argc, argv);
    auto& args_ = cli.args();

    if (args_.get(0) == "Flux2KleinPipeline_call") {
        ggml_time_init();
        ggml_log_set([](ggml_log_level, const char* text, void*) { std::cerr << text; }, nullptr);

        ggml_backend_load_all();

        auto gpus = MetaDevice::all(GGML_BACKEND_DEVICE_TYPE_GPU);
        Device cpu(GGML_BACKEND_DEVICE_TYPE_CPU);
        Backend gpus_backend(gpus);
        Backend cpu_backend(cpu);
        Scheduler scheduler({&gpus_backend, &cpu_backend}, cli.get_graph_size());

        Flux2Transformer2DModel::Config transformer_config;
        {
            transformer_config.patch_size = args_.get_optional<int64_t>("--transformer-patch_size").value_or(transformer_config.patch_size);
            transformer_config.in_channels = args_.get_optional<int64_t>("--transformer-in_channels").value_or(transformer_config.in_channels);
            transformer_config.out_channels = args_.get_optional<int64_t>("--transformer-out_channels");
            transformer_config.num_layers = args_.get_optional<int64_t>("--transformer-num_layers").value_or(transformer_config.num_layers);
            transformer_config.num_single_layers = args_.get_optional<int64_t>("--transformer-num_single_layers").value_or(transformer_config.num_single_layers);
            transformer_config.attention_head_dim = args_.get_optional<int64_t>("--transformer-attention_head_dim").value_or(transformer_config.attention_head_dim);
            transformer_config.num_attention_heads = args_.get_optional<int64_t>("--transformer-num_attention_heads").value_or(transformer_config.num_attention_heads);
            transformer_config.joint_attention_dim = args_.get_optional<int64_t>("--transformer-joint_attention_dim").value_or(transformer_config.joint_attention_dim);
            transformer_config.timestep_guidance_channels = args_.get_optional<int64_t>("--transformer-timestep_guidance_channels").value_or(transformer_config.timestep_guidance_channels);
            transformer_config.mlp_ratio = args_.get_optional<float>("--transformer-mlp_ratio").value_or(transformer_config.mlp_ratio);
            auto axes_dims_rope = args_.get_many<int64_t>("--transformer-axes_dims_rope");
            transformer_config.rope_theta = args_.get_optional<int64_t>("--transformer-rope_theta").value_or(transformer_config.rope_theta);
            transformer_config.eps = args_.get_optional<float>("--transformer-eps").value_or(transformer_config.eps);
            transformer_config.guidance_embeds = args_.get_optional<bool>("--transformer-guidance_embeds").value_or(transformer_config.guidance_embeds);

            if (!axes_dims_rope.empty())
                transformer_config.axes_dims_rope = axes_dims_rope;
        }

        AutoencoderKLFlux2::Config vae_config;
        {
            vae_config.in_channels = args_.get_optional<int64_t>("--vae-in_channels").value_or(vae_config.in_channels);
            vae_config.out_channels = args_.get_optional<int64_t>("--vae-out_channels").value_or(vae_config.out_channels);
            auto block_out_channels = args_.get_many<int64_t>("--vae-block_out_channels");
            vae_config.layers_per_block = args_.get_optional<int64_t>("--vae-layers_per_block").value_or(vae_config.layers_per_block);
            vae_config.latent_channels = args_.get_optional<int64_t>("--vae-latent_channels").value_or(vae_config.latent_channels);
            vae_config.norm_num_groups = args_.get_optional<int64_t>("--vae-norm_num_groups").value_or(vae_config.norm_num_groups);
            vae_config.sample_size = args_.get_optional<int64_t>("--vae-sample_size").value_or(vae_config.sample_size);
            vae_config.force_upcast = args_.get_optional<bool>("--vae-force_upcast").value_or(vae_config.force_upcast);
            vae_config.use_quant_conv = args_.get_optional<bool>("--vae-use_quant_conv").value_or(vae_config.use_quant_conv);
            vae_config.use_post_quant_conv = args_.get_optional<bool>("--vae-use_post_quant_conv").value_or(vae_config.use_post_quant_conv);
            vae_config.mid_block_add_attention = args_.get_optional<bool>("--vae-mid_block_add_attention").value_or(vae_config.mid_block_add_attention);
            vae_config.batch_norm_eps = args_.get_optional<float>("--vae-batch_norm_eps").value_or(vae_config.batch_norm_eps);
            vae_config.batch_norm_momentum = args_.get_optional<float>("--vae-batch_norm_momentum").value_or(vae_config.batch_norm_momentum);
            vae_config.patch_size = std::make_tuple(
                args_.get_optional<int64_t>("--vae-patch_size-0").value_or(std::get<0>(vae_config.patch_size)),
                args_.get_optional<int64_t>("--vae-patch_size-1").value_or(std::get<1>(vae_config.patch_size))
            );

            if (!block_out_channels.empty())
                vae_config.block_out_channels = block_out_channels;
        }

        Qwen3Config qwen_config;
        {
            qwen_config.vocab_size = args_.get_optional<int64_t>("--text_encoder-vocab_size").value_or(qwen_config.vocab_size);
            qwen_config.hidden_size = args_.get_optional<int64_t>("--text_encoder-hidden_size").value_or(qwen_config.hidden_size);
            qwen_config.intermediate_size = args_.get_optional<int64_t>("--text_encoder-intermediate_size").value_or(qwen_config.intermediate_size);
            qwen_config.num_hidden_layers = args_.get_optional<int64_t>("--text_encoder-num_hidden_layers").value_or(qwen_config.num_hidden_layers);
            qwen_config.num_attention_heads = args_.get_optional<int64_t>("--text_encoder-num_attention_heads").value_or(qwen_config.num_attention_heads);
            qwen_config.num_key_value_heads = args_.get_optional<int64_t>("--text_encoder-num_key_value_heads").value_or(qwen_config.num_key_value_heads);
            qwen_config.max_position_embeddings = args_.get_optional<int64_t>("--text_encoder-max_position_embeddings").value_or(qwen_config.max_position_embeddings);
        }

        auto tokenizer_dir = args_.get_one<std::string>("--tokenizer_dir");

        Context context(scheduler.capacity());
        Runtime runtime(scheduler, context);

        Flux2Transformer2DModel transformer(transformer_config);
        {
            TestCLI::CreateParametersVisitor create_parameters(runtime, args_, "transformer");
            RethrowVisitor visitor(create_parameters);
            transformer.accept(visitor);
            visitor.rethrow();
        }
        
        AutoencoderKLFlux2 vae(vae_config);
        {
            TestCLI::CreateParametersVisitor create_parameters(runtime, args_, "vae");
            RethrowVisitor visitor(create_parameters);
            vae.accept(visitor);
            visitor.rethrow();
        }

        Qwen3ForCausalLM text_encoder(qwen_config);
        {
            TestCLI::CreateParametersVisitor create_parameters(runtime, args_, "text_encoder");
            RethrowVisitor visitor(create_parameters);
            text_encoder.accept(visitor);
            visitor.rethrow();
        }

        auto tokenizer = Qwen2TokenizerFast::from_pretrained(tokenizer_dir);

        Flux2KleinPipeline pipeline(
            std::move(transformer),
            std::move(vae),
            std::move(text_encoder),
            std::move(tokenizer)
        );

        Flux2KleinPipeline::GenerationOptions options;

        options.prompt = args_.get_one<std::string>("--prompt");
        options.height = args_.get_one<int>("--height");
        options.width = args_.get_one<int>("--width");
        options.num_inference_steps = args_.get_one<int>("--num_inference_steps");
        options.max_sequence_length = args_.get_one<int>("--max_sequence_length");

        if (auto init_latents = args_.get_optional<std::string>("--init_latents"))
            options.init_latents = std::move(
                ArgumentParser::parser<Tensor>::TensorParser(*init_latents).parse().second);

        auto images = pipeline(runtime, std::move(options));
        std::vector<Tensor> results;

        for (const auto& image : images) {
            std::vector<float> pixels(image.pixels().begin(), image.pixels().end());
            cli.print_tensor_like(pixels, {(int64_t)image.height(), (int64_t)image.width(), (int64_t)image.channels()});
        }

        return EXIT_SUCCESS;
    }

    return cli.main();
}
