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
        eps_(eps),
        elementwise_affine_(elementwise_affine),
        bias_(bias)
    {
        if (elementwise_affine) {
            modules["weight"] = std::shared_ptr<Module>(new Parameter<1>({dim}));

            if (bias)
                modules["bias"] = std::shared_ptr<Module>(new Parameter<1>({dim}));
        }
    }

    Tensor forward(ggml_context* ctx, Tensor hidden_states) {
        auto variance = hidden_states.pow(2).mean(/*-1, TODO: check*/);
        
        hidden_states = hidden_states * rsqrt(variance + eps_);

        if (elementwise_affine_) {
            auto weight = std::static_pointer_cast<Parameter<1>>(modules["weight"])->forward();

            hidden_states = hidden_states * weight;

            if (bias_) {
                auto bias = std::static_pointer_cast<Parameter<1>>(modules["bias"])->forward();

                hidden_states = hidden_states + bias;
            }
        }

        return hidden_states;
    }

private:
    float eps_;
    bool elementwise_affine_, bias_;
};
