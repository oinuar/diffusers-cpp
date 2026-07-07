#pragma once

#include "nn/Module.hpp"
#include "nn/Linear.hpp"
#include "nn/SiLU.hpp"

class Qwen3MLP : public Module {
public:
    Qwen3MLP(int64_t hidden_size, int64_t intermediate_size) {
        modules["gate_proj"] = std::make_shared<Linear>(hidden_size, intermediate_size, false);
        modules["up_proj"] = std::make_shared<Linear>(hidden_size, intermediate_size, false);
        modules["down_proj"] = std::make_shared<Linear>(intermediate_size, hidden_size, false);
        modules["act_fn"] = std::make_shared<SiLU>();
    }

    Tensor forward(ggml_context* ctx, Tensor x) {
        auto gate_proj = std::static_pointer_cast<Linear>(modules["gate_proj"]);
        auto up_proj = std::static_pointer_cast<Linear>(modules["up_proj"]);
        auto down_proj = std::static_pointer_cast<Linear>(modules["down_proj"]);
        auto act_fn = std::static_pointer_cast<ActFn>(modules["act_fn"]);

        x = down_proj->forward(ctx, act_fn->forward(ctx, gate_proj->forward(ctx, x)) * up_proj->forward(ctx, x));

        return x;
    }
};
