#include "transformers/models/qwen3/Qwen3ForCausalLM.hpp"
#include "transformers/models/qwen3/Qwen3Config.hpp"
#include "transformers/models/qwen3/Qwen3Model.hpp"
#include "nn/Module.hpp"
#include "nn/Linear.hpp"
#include "nn/RethrowVisitor.hpp"
#include "ggml/GGUFLoaderVisitor.hpp"

Qwen3ForCausalLM Qwen3ForCausalLM::from_pretrained(Runtime& runtime, Qwen3Config&& config, const std::filesystem::path& path) {
    Qwen3ForCausalLM model(config);

    GGUFLoaderVisitor loader(runtime, path);
    RethrowVisitor visitor(loader);

    model.accept(visitor);
    visitor.rethrow();

    return std::move(model);
}

Qwen3ForCausalLM::Qwen3ForCausalLM(const Qwen3Config& config) {
    this->vocab_size = config.vocab_size;
    modules["model"] = std::make_shared<Qwen3Model>(config);
    modules["lm_head"] = std::make_shared<Linear>(config.hidden_size, config.vocab_size, false);
}

Tensor Qwen3ForCausalLM::forward(
    Runtime& runtime, 
    std::optional<Tensor> input_ids, 
    std::optional<Tensor> attention_mask,
    std::optional<Tensor> position_ids,
    std::optional<Tensor> past_key_values,
    std::optional<Tensor> inputs_embeds,
    std::optional<Tensor> labels,
    std::optional<bool> use_cache,
    int logits_to_keep,
    std::vector<Tensor>* extract_hidden_states
) {
    auto model = std::static_pointer_cast<Qwen3Model>(modules["model"]);
    auto hidden_states = model->forward(
        runtime,
        input_ids,
        inputs_embeds,
        attention_mask,
        position_ids,
        past_key_values,
        use_cache,
        extract_hidden_states
    );

    auto lm_head = std::static_pointer_cast<Linear>(modules["lm_head"]);
    
    Tensor slice_hidden_states = hidden_states;
    if (logits_to_keep > 0) {
        int seq_len = hidden_states.shape()[1];
        slice_hidden_states = hidden_states[{Tensor::Slice::all(), Tensor::Slice::range(seq_len - logits_to_keep, seq_len)}];
    }

    auto logits = lm_head->forward(runtime, slice_hidden_states);

    // Note: Loss computation is omitted as per typical C++ porting of inference-only modules, 
    // but the primary output (logits) is returned to match the specification.
    return logits;
}
