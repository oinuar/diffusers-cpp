#include "models/transformers/qwen3/Qwen3ForCausalLM.hpp"
#include "models/transformers/qwen3/Qwen3Model.hpp"

Qwen3ForCausalLM::Qwen3ForCausalLM(const Qwen3Config& config) {
    modules["model"] = std::make_shared<Qwen3Model>(config);
    modules["lm_head"] = std::make_shared<Linear>(config.hidden_size, config.vocab_size, false);

    // TODO: post_init? what does it do?
}

Qwen3CausalLMOutput Qwen3ForCausalLM::forward(
    ggml_context* ctx,
    std::optional<Tensor> input_ids,
    std::optional<Tensor> attention_mask,
    std::optional<Tensor> position_ids,
    std::optional<Tensor> inputs_embeds,
    bool use_cache,
    int logits_to_keep)
{
    Qwen3Cache past_key_values;

    auto model = std::static_pointer_cast<Qwen3Model>(modules["model"]);

    auto outputs = model->forward(
        ctx,
        input_ids,
        attention_mask,
        position_ids,
        past_key_values,
        inputs_embeds,
        use_cache);

    hidden_states = outputs.last_hidden_state;

    auto lm_head = std::static_pointer_cast<Linear>(modules["lm_head"]);
    auto logits = lm_head->forward(ctx, hidden_states[{Tensor::Slice::all(), Tensor::Slice::range(-logits_to_keep, std::nullopt), Tensor::Slice::all()}]);

    // TODO: loss function? maybe not needed

    return CausalLMOutputWithPast{
        .loss = std::nullopt,
        .logits = logits,
        .past_key_values = outputs.past_key_values,
        .hidden_states = outputs.hidden_states,
        .attentions = outputs.attentions
    };
}
