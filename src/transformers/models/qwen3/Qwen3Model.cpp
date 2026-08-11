#include "transformers/models/qwen3/Qwen3Model.hpp"
#include "transformers/models/qwen3/Qwen3Config.hpp"
#include "transformers/models/qwen3/Qwen3DecoderLayer.hpp"
#include "transformers/models/qwen3/Qwen3RMSNorm.hpp"
#include "transformers/models/qwen3/Qwen3RotaryEmbedding.hpp"
#include "nn/ModuleList.hpp"
#include "nn/Embedding.hpp"
#include "ggml/Runtime.hpp"
#include <limits>

Qwen3Model::Qwen3Model(const Qwen3Config& config) {
    this->vocab_size = config.vocab_size;
    this->num_hidden_layers = config.num_hidden_layers;
    this->layer_types = config.layer_types;
    this->has_sliding_layers = false;
    this->sliding_window = config.sliding_window;

    modules["embed_tokens"] = std::make_shared<Embedding>(config.vocab_size, config.hidden_size, config.pad_token_id);

    auto layers = std::make_shared<ModuleList>(config.num_hidden_layers);
    modules["layers"] = layers;

    for (int i = 0; i < layers->size(); ++i) {
        (*layers)[i] = std::make_shared<Qwen3DecoderLayer>(config, i);
        this->has_sliding_layers |= this->layer_types.at(i) == "sliding_attention";
    }

    modules["norm"] = std::make_shared<Qwen3RMSNorm>(config.hidden_size, config.rms_norm_eps);

    // We put rotary_emb here to match Python module structure, and pass it
    // down to lower modules where it is actually used
    modules["rotary_emb"] = std::make_shared<Qwen3RotaryEmbedding>(config);
}

static Tensor create_causal_mask(
    Runtime& runtime,
    int seq_len,
    int target_len,
    int past_seen_tokens,
    std::optional<Tensor> attention_mask,
    std::optional<int> sliding_window = std::nullopt
) {
    const float neg_inf = -std::numeric_limits<float>::infinity();

    auto mask = runtime.create<float>({1, 1, seq_len, target_len},
        [=](std::mt19937&) {
            std::vector<float> mask;
            mask.reserve(seq_len * target_len);

            for (int i = 0; i < seq_len; ++i) {
                for (int j = 0; j < target_len; ++j) {
                    bool allowed =
                        j <= i + past_seen_tokens;

                    if (sliding_window)
                        allowed =
                            allowed &&
                            (j > i + past_seen_tokens - *sliding_window);

                    mask.push_back(
                        allowed ? 0.0f : neg_inf
                    );
                }
            }

            return mask;
        }
    );

    if (attention_mask) {
        auto expanded_mask = attention_mask.value()
            .unsqueeze(1)
            .unsqueeze(1);

        // 1 -> 0
        // 0 -> large negative value.
        //
        // Do NOT use -inf here:
        // (1 - 1) * -inf == 0 * -inf == NaN
        expanded_mask =
            (1.0f - expanded_mask) * -1e9f;

        expanded_mask = expanded_mask.expand({
            attention_mask.value().shape()[0],
            1,
            seq_len,
            target_len
        });

        mask = mask + expanded_mask;
    }

    return mask;
}

Tensor Qwen3Model::forward(
    Runtime& runtime,
    std::optional<Tensor> input_ids,
    std::optional<Tensor> inputs_embeds,
    std::optional<Tensor> attention_mask,
    std::optional<Tensor> position_ids,
    std::optional<Tensor> past_key_values,
    std::optional<bool> use_cache,
    std::vector<Tensor>* extract_hidden_states
) {
    if (!input_ids.has_value() && !inputs_embeds.has_value())
        throw std::invalid_argument(
            "You must specify exactly one of input_ids or inputs_embeds"
        );

    if (!inputs_embeds) {
        auto embed_tokens =
            std::static_pointer_cast<Embedding>(modules["embed_tokens"]);

        inputs_embeds = embed_tokens->forward(runtime, input_ids.value());
    }

    auto hidden_states = *inputs_embeds;

    auto seq_len = hidden_states.shape()[1];

    // TODO: replace with past_key_values length when cache is implemented
    auto past_seen_tokens = 0;

    if (!position_ids) {
        position_ids = Tensor::arange(*runtime.context(), 0, seq_len);
        position_ids = *position_ids + (float)past_seen_tokens;
        position_ids = position_ids.value().unsqueeze(0);
    }

    auto rotary_emb =
        std::static_pointer_cast<Qwen3RotaryEmbedding>(modules["rotary_emb"]);

    auto target_len = past_seen_tokens + seq_len;

    std::unordered_map<std::string, Tensor> causal_mask_mapping;

    causal_mask_mapping["full_attention"] = create_causal_mask(
        runtime,
        seq_len,
        target_len,
        past_seen_tokens,
        attention_mask
    );

    if (has_sliding_layers) {
        causal_mask_mapping["sliding_attention"] = create_causal_mask(
            runtime,
            seq_len,
            target_len,
            past_seen_tokens,
            attention_mask,
            sliding_window
        );
    }

    auto layers = std::static_pointer_cast<ModuleList>(modules["layers"]);

    for (auto i = 0; i < layers->size(); ++i) {
        auto layer =
            std::static_pointer_cast<Qwen3DecoderLayer>((*layers)[i]);

        auto layer_mask = causal_mask_mapping.at(layer_types.at(i));

        if (extract_hidden_states)
            extract_hidden_states->push_back(hidden_states.clone());

        hidden_states = layer->forward(
            runtime,
            *rotary_emb,
            hidden_states,
            *position_ids,
            layer_mask,
            past_key_values,
            use_cache
        );
    }

    auto norm =
        std::static_pointer_cast<Qwen3RMSNorm>(modules["norm"]);

    hidden_states = norm->forward(runtime, hidden_states);

    if (extract_hidden_states)
        extract_hidden_states->push_back(hidden_states);

    return hidden_states;
}
