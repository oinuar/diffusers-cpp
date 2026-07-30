#include "transformers/models/qwen3/Qwen3MLP.hpp"
#include "transformers/models/qwen3/Qwen3Config.hpp"
#include "nn/Linear.hpp"
#include "nn/SiLU.hpp"

Qwen3MLP::Qwen3MLP(const Qwen3Config& config) {
    modules["gate_proj"] = std::make_shared<Linear>(config.hidden_size, config.intermediate_size, config.attention_bias);
    modules["up_proj"] = std::make_shared<Linear>(config.hidden_size, config.intermediate_size, config.attention_bias);
    modules["down_proj"] = std::make_shared<Linear>(config.intermediate_size, config.hidden_size, config.attention_bias);
    modules["act_fn"] = std::make_shared<SiLU>();
}

Tensor Qwen3MLP::forward(Runtime& runtime, Tensor x) {
    auto gate_proj = std::static_pointer_cast<Linear>(modules["gate_proj"]);
    auto up_proj = std::static_pointer_cast<Linear>(modules["up_proj"]);
    auto down_proj = std::static_pointer_cast<Linear>(modules["down_proj"]);
    auto act_fn = std::static_pointer_cast<SiLU>(modules["act_fn"]);

    auto gate = gate_proj->forward(runtime, x);
    auto up = up_proj->forward(runtime, x);
    
    gate = act_fn->forward(runtime, gate);

    return down_proj->forward(runtime, gate * up);
}
