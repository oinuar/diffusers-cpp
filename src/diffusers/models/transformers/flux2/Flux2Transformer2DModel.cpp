#include "diffusers/models/transformers/flux2/Flux2Transformer2DModel.hpp"
#include "diffusers/models/transformers/flux2/Flux2PosEmbed.hpp"
#include "diffusers/models/transformers/flux2/Flux2TimestepGuidanceEmbeddings.hpp"
#include "diffusers/models/transformers/flux2/Flux2Modulation.hpp"
#include "diffusers/models/transformers/flux2/Flux2TransformerBlock.hpp"
#include "diffusers/models/transformers/flux2/Flux2SingleTransformerBlock.hpp"
#include "diffusers/models/normalization/AdaLayerNormContinuous.hpp"
#include "nn/Linear.hpp"
#include "nn/attention/ScaledDotProductAttention.hpp"
#include "nn/attention/FlashAttentionOp.hpp"
#include "ggml/GGUFLoaderVisitor.hpp"

Flux2Transformer2DModel Flux2Transformer2DModel::from_pretrained(Backend& loader_backend, const std::string& path) {
    Flux2Transformer2DModel model(
        1,
        128,
        std::nullopt,
        8,
        24,
        128,
        32,
        12288,
        256,
        3.0,
        {32, 32, 32, 32},
        2000,
        1e-6,
        false
    );

    GGUFLoaderVisitor loader(loader_backend, path);
    model.accept(loader);

    return std::move(model);
}

Flux2Transformer2DModel::Flux2Transformer2DModel(
    int64_t patch_size,
    int64_t in_channels,
    std::optional<int64_t> out_channels,
    int64_t num_layers,
    int64_t num_single_layers,
    int64_t attention_head_dim,
    int64_t num_attention_heads,
    int64_t joint_attention_dim,
    int64_t timestep_guidance_channels,
    float mlp_ratio,
    const std::vector<int64_t>& axes_dims_rope,
    int64_t rope_theta,
    float eps,
    bool guidance_embeds
) {
    auto inner_dim = num_attention_heads * attention_head_dim;

    // 1. Sinusoidal positional embedding for RoPE on image and text tokens
    modules["pos_embed"] = std::make_shared<Flux2PosEmbed>(rope_theta, axes_dims_rope);

    // 2. Combined timestep + guidance embedding
    modules["time_guidance_embed"] = std::make_shared<Flux2TimestepGuidanceEmbeddings>(
        timestep_guidance_channels,
        inner_dim,
        false,
        guidance_embeds
    );

    // 3. Modulation (double stream and single stream blocks share modulation parameters, resp.)
    // Two sets of shift/scale/gate modulation parameters for the double stream attn and FF sub-blocks
    modules["double_stream_modulation_img"] = std::make_shared<Flux2Modulation>(inner_dim, 2, false);
    modules["double_stream_modulation_txt"] = std::make_shared<Flux2Modulation>(inner_dim, 2, false);
    // Only one set of modulation parameters as the attn and FF sub-blocks are run in parallel for single stream
    modules["single_stream_modulation"] = std::make_shared<Flux2Modulation>(inner_dim, 1, false);

    // 4. Input projections
    modules["x_embedder"] = std::make_shared<Linear>(in_channels, inner_dim, false);
    modules["context_embedder"] = std::make_shared<Linear>(joint_attention_dim, inner_dim, false);

    // 5. Double Stream Transformer Blocks
    auto transformer_blocks = std::make_shared<ModuleList>(num_layers);
    modules["transformer_blocks"] = transformer_blocks;

    for (auto i = 0; i < transformer_blocks->size(); ++i)
        (*transformer_blocks)[i] =
            std::make_shared<Flux2TransformerBlock<ScaledDotProductAttention<FlashAttentionOp>>>(
                inner_dim,
                num_attention_heads,
                attention_head_dim,
                mlp_ratio,
                eps,
                false
            );


    // 6. Single Stream Transformer Blocks
    auto single_transformer_blocks = std::make_shared<ModuleList>(num_single_layers);
    modules["single_transformer_blocks"] = single_transformer_blocks;

    for (auto i = 0; i < single_transformer_blocks->size(); ++i)
        (*single_transformer_blocks)[i] =
            std::make_shared<Flux2SingleTransformerBlock<ScaledDotProductAttention<FlashAttentionOp>>>(
                inner_dim,
                num_attention_heads,
                attention_head_dim,
                mlp_ratio,
                eps,
                false
            );

    // 7. Output layers
    modules["norm_out"] = std::make_shared<AdaLayerNormContinuous<>>(inner_dim, inner_dim, false, eps, false);
    modules["proj_out"] = std::make_shared<Linear>(inner_dim, patch_size * patch_size * out_channels.value_or(in_channels), false);
}

Tensor Flux2Transformer2DModel::forward(
    Runtime& runtime,
    Tensor hidden_states,
    Tensor encoder_hidden_states,
    Tensor timestep,
    Tensor img_ids,
    Tensor txt_ids,
    std::optional<Tensor> guidance,
    // TODO: support KV cache
    //Tensor joint_attention_kwargs: dict[str, Any] | None = None,
    //Tensor return_dict: bool = True,
    //Tensor kv_cache: "Flux2KVCache | None" = None,
    //Tensor kv_cache_mode: str | None = None,
    int64_t num_ref_tokens,
    float ref_fixed_timestep
) {
    auto num_txt_tokens = encoder_hidden_states.shape()[1];

    // 1. Calculate timestep embedding and modulation parameters
    timestep = timestep.to(hidden_states.dtype()) * 1000;

    if (guidance)
        guidance = guidance->to(hidden_states.dtype()) * 1000;

    auto time_guidance_embed = std::static_pointer_cast<Flux2TimestepGuidanceEmbeddings>(modules["time_guidance_embed"]);

    auto temb = time_guidance_embed->forward(runtime, timestep, guidance);

    auto double_stream_modulation_img = std::static_pointer_cast<Flux2Modulation>(modules["double_stream_modulation_img"]);
    auto double_stream_modulation_txt = std::static_pointer_cast<Flux2Modulation>(modules["double_stream_modulation_txt"]);
    auto single_stream_modulation = std::static_pointer_cast<Flux2Modulation>(modules["single_stream_modulation"]);

    auto double_stream_mod_img = double_stream_modulation_img->forward(runtime, temb);
    auto double_stream_mod_txt = double_stream_modulation_txt->forward(runtime, temb);
    auto single_stream_mod = single_stream_modulation->forward(runtime, temb);

    // TODO: KV extract mode: create cache and blend modulations for ref tokens

    // 2. Input projection for image (hidden_states) and conditioning text (encoder_hidden_states)
    auto x_embedder = std::static_pointer_cast<Linear>(modules["x_embedder"]);
    auto context_embedder = std::static_pointer_cast<Linear>(modules["context_embedder"]);

    hidden_states = x_embedder->forward(runtime, hidden_states);
    encoder_hidden_states = context_embedder->forward(runtime, encoder_hidden_states);

    // 3. Calculate RoPE embeddings from image and text tokens
    if (img_ids.ndim() == 3)
        img_ids = img_ids[0];
    if (txt_ids.ndim() == 3)
        txt_ids = txt_ids[0];

    // This is different from Python implementation because we'll compute RoPE directly using position IDs,
    // so embeddings setup is like this:
    auto concat_rotary_emb = std::make_pair(
        std::static_pointer_cast<Flux2PosEmbed>(modules["pos_embed"]),

        // [Nt, 4] + [Ni, 4] -> [Nt + Ni, 4]
        Tensor::cat({txt_ids, img_ids}, 0)
    );

    // TODO: 4. Build joint_attention_kwargs with KV cache info

    // 5. Double Stream Transformer Blocks
    auto transformer_blocks = std::static_pointer_cast<ModuleList>(modules["transformer_blocks"]);
    
    for (auto i = 0; i < transformer_blocks->size(); ++i) {
        auto block = std::static_pointer_cast<Flux2TransformerBlock<ScaledDotProductAttention<FlashAttentionOp>>>(
            (*transformer_blocks)[i]
        );

        auto [fwd_encoder_hidden_states, fwd_hidden_states] = block->forward(
            runtime,
            hidden_states,
            encoder_hidden_states,
            double_stream_mod_img,
            double_stream_mod_txt,
            concat_rotary_emb
        );

        encoder_hidden_states = fwd_encoder_hidden_states;
        hidden_states = fwd_hidden_states;
    }

    // Concatenate text and image streams for single-block inference
    hidden_states = Tensor::cat({encoder_hidden_states, hidden_states}, 1);

    // TODO: Blend single block modulation for extract mode: [txt_mod, ref_mod, img_mod]
    // TOOD: Build single-block KV kwargs (single blocks need num_txt_tokens)

    // 6. Single Stream Transformer Blocks
    auto single_transformer_blocks = std::static_pointer_cast<ModuleList>(modules["single_transformer_blocks"]);

    for (auto i = 0; i < single_transformer_blocks->size(); ++i) {
        auto block = std::static_pointer_cast<Flux2SingleTransformerBlock<ScaledDotProductAttention<FlashAttentionOp>>>(
            (*single_transformer_blocks)[i]
        );

        auto [fwd_hidden_states, _] = block->forward(
            runtime,
            hidden_states,
            std::nullopt,
            single_stream_mod,
            concat_rotary_emb
        );

        hidden_states = fwd_hidden_states;
    }

    // TODO: Remove text tokens (and ref tokens in extract mode) from concatenated stream

    hidden_states = hidden_states[{Tensor::Slice::all(), Tensor::Slice::range(num_txt_tokens, std::nullopt), Tensor::Slice::ellipsis()}];

    // 7. Output layers
    auto norm_out = std::static_pointer_cast<AdaLayerNormContinuous<>>(modules["norm_out"]);
    auto proj_out = std::static_pointer_cast<Linear>(modules["proj_out"]);

    hidden_states = norm_out->forward(runtime, hidden_states, temb);
    auto output = proj_out->forward(runtime, hidden_states);

    return output;
}