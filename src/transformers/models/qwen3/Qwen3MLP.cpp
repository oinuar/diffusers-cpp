#include "transformers/models/qwen3/Qwen3MLP.hpp"
#include "transformers/models/qwen3/Qwen3Config.hpp"
#include "nn/Linear.hpp"
#include "nn/SiLU.hpp"

Qwen3MLP::Qwen3MLP(const Qwen3Config& config) {
    modules["gate_proj"] = std::make_shared<Linear>(config.hidden_size, config.intermediate_size, false);
    modules["up_proj"] = std::make_shared<Linear>(config.hidden_size, config.intermediate_size, false);
    modules["down_proj"] = std::make_shared<Linear>(config.intermediate_size, config.hidden_size, false);
    modules["act_fn"] = std::make_shared<SiLU>();
}

Tensor Qwen3MLP::forward(Scope scope, Tensor x) {
    auto gate_proj = std::static_pointer_cast<Linear>(modules["gate_proj"]);
    auto up_proj = std::static_pointer_cast<Linear>(modules["up_proj"]);
    auto down_proj = std::static_pointer_cast<Linear>(modules["down_proj"]);
    auto act_fn = std::static_pointer_cast<SiLU>(modules["act_fn"]);

    auto gate = gate_proj->forward(scope, x);
    auto up = up_proj->forward(scope, x);
    
    gate = act_fn->forward(scope, gate);

    return down_proj->forward(scope, gate * up);
}
