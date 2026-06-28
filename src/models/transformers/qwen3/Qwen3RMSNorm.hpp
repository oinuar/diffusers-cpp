#pragma once

#include "modules/Module.hpp"
#include "modules/Parameter.hpp"

class Qwen3RMSNorm : public Module {
public:
    Qwen3RMSNorm(int64_t dim, float eps = 1e-6f)
        : dim_(dim), variance_epsilon_(eps)
    {
        modules["weight"] = std::make_shared<Parameter>(Tensor::Shape({dim}));
    }

    Tensor forward(ggml_context* ctx, Tensor hidden_states) {
        auto input_dtype = hidden_states.dtype();

        hidden_states = hidden_states.to(GGML_TYPE_F32);

        auto variance = hidden_states.pow(2).mean(/* TODO: -1, keepdim=True*/);

        hidden_states = hidden_states * rsqrt(variance + variance_epsilon_);

        auto weight = std::static_pointer_cast<Parameter>(modules["weight"])->forward();
        return weight * hidden_states.to(input_dtype);
    }

private:
    int64_t dim_;
    float variance_epsilon_;
};
