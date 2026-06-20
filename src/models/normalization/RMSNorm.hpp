#pragma once

#include "modules/Module.hpp"

class RMSNorm : public Module {
public:
    RMSNorm(
        int64_t dim,
        float eps = 1e-5f,
        bool elementwise_affine = true,
        bool bias = false
    ) : 
        eps_(eps)
    {
        if (elementwise_affine) {
            params["weight"] = Parameter([=](ggml_context* ctx) { return Tensor::ones<1>(ctx, {dim}); });

            if (bias)
                params["bias"] = Parameter([=](ggml_context* ctx) { return Tensor::zeros<1>(ctx, {dim}); });
        }
    }

    Tensor forward(ggml_context* ctx, Tensor hidden_states) {
        auto weight = params.find("weight");
        auto bias = params.find("bias");
        auto variance = hidden_states.pow(2).mean(/*-1, TODO: check*/);
        
        hidden_states = hidden_states * rsqrt(variance + eps_);

        if (weight != params.end()) {
            hidden_states = hidden_states * *weight->second;

            if (bias != params.end())
                hidden_states = hidden_states + *bias->second;
        }

        return hidden_states;
    }

private:
    float eps_;
};
