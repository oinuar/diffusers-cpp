Qwen3ForCausalLM::Qwen3ForCausalLM(const Qwen3Config& config) {
    modules["model"] = std::make_shared<Qwen3Model>(config);
    modules["lm_head"] = std::make_shared<Linear>(config.hidden_size, config.vocab_size, false);

    //# Initialize weights and apply final processing
    //self.post_init()
}

Tensor Qwen3ForCausalLM::forward(
    ggml_context* ctx,
    std::optional<std::vector<int64_t>> input_ids,
    std::optional<Tensor> attention_mask,
    std::optional<std::vector<int64_t>> position_ids,
    // past_key_values: Cache | None = None,
    std::optional<Tensor> inputs_embeds,
    std::optional<std::vector<int64_t>> labels,
    bool use_cache,
    int logits_to_keep
    //**kwargs: Unpack[TransformersKwargs],
)
{
    if (labels)
        throw std::runtime_error("labels are not supported");

    auto model = std::static_pointer_cast<Qwen3Model>(modules["model"]);

    auto hidden_states = model->forward(
        ctx,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        **kwargs,
    )

    auto lm_head = std::static_pointer_cast<Linear>(modules["lm_head"]);

    // Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    logits = lm_head->forward(ctx, hidden_states[{Tensor::Slice::all(), Tensor::Slice::Range(-logits_to_keep, std::nullopt), Tensor::Slice::all()}])

    //loss = None
    //if labels is not None:
        //loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    );
}




/*#include "models/transformers/qwen3/Qwen3Model.hpp"
#include "models/transformers/qwen3/Qwen3DecoderLayer.hpp"
#include "models/embeddings/Qwen3RotaryEmbedding.hpp"
#include "models/normalization/RMSNorm.hpp"

Qwen3Model::Qwen3Model(const Config& config) : config_(config) {
    // Register layers based on their types.
    for (int i = 0; i < config_.num_hidden_layers; ++i) {
        std::string layer_type = config_.layer_types.empty() ? "full_attention" : config_.layer_types[i];
        if (layer_type.empty()) layer_type = "full_attention";

        auto layer_config = Qwen3DecoderLayer::Config(
            config_.hidden_size, config_.intermediate_size, config_.rms_norm_eps,
            config_.num_attention_heads, config_.num_key_value_heads);

        modules["layers." + std::to_string(i)] = std::make_shared<Qwen3DecoderLayer>(layer_config, layer_type);
    }

    // Rotary embedding (computed from config, not loaded from GGUF).
    modules["rotary_emb"] = std::make_shared<Qwen3RotaryEmbedding>(
        config_.rope_theta, config_.hidden_size / config_.num_attention_heads, config_.max_position_embeddings);

    // Final RMSNorm (full-dimension).
    modules["norm"] = std::make_shared<RMSNorm>(config_.hidden_size, config_.rms_norm_eps);

    // Embedding weight parameter (loaded from GGUF).
    modules["embed_tokens.weight"] = std::make_shared<Parameter>(Tensor::Shape({config_.vocab_size, config_.hidden_size}));
}

Tensor Qwen3Model::create_causal_mask(ggml_context* ctx, int batch, int seq_len) {
    // Create upper triangular mask of -inf (or large negative value).
    // Shape: [batch, 1, seq_len, seq_len].
    auto ones = Tensor::ones(ctx, Tensor::Shape({seq_len}));
    auto mask = Tensor::zeros(ctx, Tensor::Shape({seq_len}));

    // Fill upper triangle with -inf. Use a large negative value instead of actual -inf.
    const float neg_inf = -1e9f;

    // Build mask using a [seq_len, seq_len] matrix where mask[i,j] = -inf if j > i, else 0.
    auto row_indices = Tensor::arange(ctx, 0.0f, static_cast<float>(seq_len));
    auto col_indices = Tensor::arange(ctx, 0.0f, static_cast<float>(seq_len));

    // Expand for broadcasting: [seq_len, 1] and [1, seq_len].
    auto rows = row_indices.unsqueeze(1);  // [S, 1]
    auto cols = col_indices.unsqueeze(0);   // [1, S]

    // mask[i,j] = -inf if j > i.
    auto causal_mask = (cols > rows) * neg_inf;  // Broadcasting: [S, S]

    // Expand to [1, 1, S, S].
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0);

    // Broadcast to batch size.
    if (batch > 1) {
        std::vector<Tensor> batch_masks;
        for (int b = 0; b < batch; ++b) {
            batch_masks.push_back(causal_mask);
        }
        causal_mask = Tensor::cat(batch_masks, 0); // [B, 1, S, S]
    }

    return causal_mask;
}

Tensor Qwen3Model::create_sliding_window_causal_mask(ggml_context* ctx, int batch, int seq_len) {
    // Same as full causal mask but with additional band constraint.
    // Only allow attention within sliding_window tokens.
    auto rows = Tensor::arange(ctx, 0.0f, static_cast<float>(seq_len)).unsqueeze(1);  // [S, 1]
    auto cols = Tensor::arange(ctx, 0.0f, static_cast<float>(seq_len)).unsqueeze(0);   // [1, S]

    const float neg_inf = -1e9f;
    // mask[i,j] = -inf if j > i OR (i - j) > sliding_window.
    auto causal      = (cols > rows) * neg_inf;
    auto window_mask = ((rows - cols) > static_cast<float>(config_.sliding_window)) * neg_inf;

    auto mask = Tensor::cat({causal, window_mask}, 0); // [2, S, S]
    mask = Tensor::max(mask, 0);                        // Element-wise max → [S, S] (takes the less negative of the two)

    mask = mask.unsqueeze(0).unsqueeze(0);

    if (batch > 1) {
        std::vector<Tensor> batch_masks;
        for (int b = 0; b < batch; ++b) batch_masks.push_back(mask);
        mask = Tensor::cat(batch_masks, 0); // [B, 1, S, S]
    }

    return mask;
}

std::tuple<Tensor, std::shared_ptr<Qwen3Cache>> Qwen3Model::forward(
    ggml_context* ctx,
    Tensor input_ids,
    std::optional<Tensor> attention_mask,
    std::optional<Tensor> position_ids,
    std::shared_ptr<Qwen3Cache> past_key_values)
{
    const int64_t batch  = input_ids.shape()[0];
    const int64_t seq_len = input_ids.shape()[1];

    // 1. Embedding lookup: inputs_embeds = gather(embed_tokens.weight, input_ids).
    // Using ggml_get_rows equivalent via narrow + cat pattern or a custom implementation.
    auto embed_weight = std::static_pointer_cast<Parameter>(modules["embed_tokens.weight"])->forward();
    Tensor inputs_embeds;

    // For now, use operator[] with slicing to gather rows. This requires input_ids to be processed element by element.
    // A more efficient implementation would use ggml_get_rows directly.
    std::vector<Tensor> embedded_rows;
    for (int64_t i = 0; i < seq_len; ++i) {
        auto row_idx = input_ids[i].squeeze(0);  // Get scalar index.
        // Gather row: embed_weight[row_idx] — needs a gather operation.
        // Placeholder: use narrow + squeeze pattern for each unique index.
        // This is O(seq_len * vocab_size) naive approach — will be optimized with ggml_get_rows.
    }

    // TODO: Implement efficient embedding lookup via ggml_get_rows or custom gather op.
    // For now, return placeholder result until embedding lookup is implemented.

    throw std::runtime_error("Qwen3Model::forward(input_ids): embedding lookup not yet implemented. "
                             "Use forward_with_embeddings() with pre-computed embeddings instead.");
}

std::tuple<Tensor, Tensor, std::shared_ptr<Qwen3Cache>> Qwen3Model::forward_with_embeddings(
    ggml_context* ctx,
    Tensor inputs_embeds,
    std::optional<Tensor> attention_mask,
    std::optional<Tensor> position_ids,
    std::shared_ptr<Qwen3Cache> past_key_values)
{
    const int64_t batch  = inputs_embeds.shape()[0];
    const int64_t seq_len = inputs_embeds.shape()[1];

    // 2. Position IDs: auto-compute if not provided.
    if (!position_ids.has_value()) {
        int64_t past_len = past_key_values ? past_key_values->get_seq_length() : 0;
        position_ids = Tensor::arange(ctx, static_cast<float>(past_len), static_cast<float>(past_len + seq_len));
        position_ids = position_ids.unsqueeze(0); // [1, seq_len]
    }

    // 3. Build causal mask based on layer types.
    std::optional<Tensor> full_attention_mask;
    bool has_sliding = false;

    for (int i = 0; i < config_.num_hidden_layers; ++i) {
        std::string layer_type = config_.layer_types.empty() ? "full_attention" : config_.layer_types[i];

        if (layer_type == "sliding_window") {
            has_sliding = true;
            break;
        }
    }

    // Build rotary embeddings once (shared across all layers).
    auto& rotary_emb = std::static_pointer_cast<Qwen3RotaryEmbedding>(modules["rotary_emb"]);
    auto [cos, sin]  = rotary_emb->forward(ctx, inputs_embeds, position_ids.value());
    auto rotary_pair = std::make_pair(cos, sin);

    // 4. Forward through layers.
    for (int i = 0; i < config_.num_hidden_layers; ++i) {
        std::string layer_type = config_.layer_types.empty() ? "full_attention" : config_.layer_types[i];
        if (layer_type.empty()) layer_type = "full_attention";

        // Build per-layer causal mask.
        Tensor layer_mask;
        if (layer_type == "sliding_window") {
            layer_mask = create_sliding_window_causal_mask(ctx, batch, seq_len);
        } else {
            layer_mask = create_causal_mask(ctx, batch, seq_len);
        }

        auto& decoder_layer = std::static_pointer_cast<Qwen3DecoderLayer>(modules["layers." + std::to_string(i)]);
        inputs_embeds = decoder_layer->forward(
            ctx, inputs_embeds, rotary_pair, layer_mask, past_key_values, i);
    }

    // 5. Final RMSNorm.
    auto last_hidden_state = std::static_pointer_cast<RMSNorm>(modules["norm"])->forward(ctx, inputs_embeds);

    return {last_hidden_state, position_ids.value(), past_key_values};
}
*/