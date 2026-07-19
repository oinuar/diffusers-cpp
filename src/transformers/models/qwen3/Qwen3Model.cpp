#include "models/transformers/qwen3/Qwen3Model.hpp"
#include "models/transformers/qwen3/Qwen3DecoderLayer.hpp"
#include "models/transformers/qwen3/Qwen3RotaryEmbedding.hpp"
#include "nn/Embedding.hpp"

Qwen3Model::Qwen3Model(const Qwen3Config& config) : num_hidden_layers_(config.num_hidden_layers) {
    modules["embed_tokens"] = std::make_shared<Embedding>(config.vocab_size, config.hidden_size, config.pad_token_id);

    for (int i = 0; i < config.num_hidden_layers; ++i)
        modules["layers." + std::to_string(i)] = std::make_shared<Qwen3DecoderLayer>(config, i);

    modules["norm"] = std::make_shared<Qwen3RMSNorm>(config.hidden_size, config.rms_norm_eps);
    modules["rotary_emb"] = std::make_shared<Qwen3RotaryEmbedding>(config);

    // TODO: post_init? what does it do?
}

BaseModelOutputWithPast Qwen3Model::forward(
    ggml_context* ctx,
    std::optional<Tensor> input_ids,
    std::optional<Tensor> attention_mask,
    std::optional<Tensor> position_ids,
    std::optional<Qwen3Cache> past_key_values,
    std::optional<Tensor> inputs_embeds,
    bool use_cache = false)
{
    if (!input_ids && !inputs_embeds)
        throw std::invalid_argument("Must provide either input_ids or inputs_embeds");

    if (!inputs_embeds) {
        auto embed_tokens = std::static_pointer_cast<Embedding>(modules["embed_tokens"]);
        inputs_embeds = embed_tokens->forward(ctx, input_ids.value());
    }

    if (use_cache && !past_key_values)
        past_key_values = Qwen3Cache();

    if (!position_ids) {
        auto past_seen_tokens = past_key_values ? past_key_values->get_seq_length() : 0;
        position_ids = Tensor::arange(inputs_embeds.shape()[1] + past_seen_tokens);
        position_ids = position_ids.unsqueeze(0);
    }

    // TODO: not isinstance(causal_mask_mapping := attention_mask, dict): whole block of attention mask creation is missing

    hidden_states = inputs_embeds.value();

    auto rotary_emb = std::static_pointer_cast<Qwen3RotaryEmbedding>(modules["rotary_emb"]);
    auto position_embeddings = rotary_emb->forward(ctx, hidden_states, position_ids);

    for (auto i = 0; i < num_hidden_layers_; ++i) {
        auto decoder_layer = std::static_pointer_cast<Qwen3DecoderLayer>(modules["layers." + std::to_string(i)]);

        hidden_states = decoder_layer->forward(
            ctx,
            hidden_states,
            attention_mask,
            position_embeddings,
            past_key_values,
            use_cache);
    }

    auto norm = std::static_pointer_cast<Qwen3RMSNorm>(modules["norm"]);
    hidden_states = norm->forward(ctx, hidden_states);

    return BaseModelOutputWithPast{
        .last_hidden_state = hidden_states,
        .past_key_values = use_cache ? std::move(past_key_values.value()) : std::nullopt
    };
}
