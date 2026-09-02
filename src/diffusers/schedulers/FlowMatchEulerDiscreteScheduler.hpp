#pragma once

#include <optional>
#include <string>

#include "diffusers/schedulers/Schedule.hpp"
#include "ggml/Scope.hpp"

class FlowMatchEulerDiscreteScheduler {
public:
    FlowMatchEulerDiscreteScheduler(
        int num_train_timesteps = 1000,
        float shift = 1.0f,
        bool use_dynamic_shifting = false,
        std::optional<float> shift_terminal = std::nullopt,
        bool invert_sigmas = false,
        std::string time_shift_type = "exponential"
    );

    Schedule schedule(
        int num_inference_steps,
        float mu = 0.0f,
        std::optional<std::vector<float>> sigmas = std::nullopt
    ) const;

    Tensor integrate(
        Scope scope,
        Tensor model_output,
        Tensor sample,
        Tensor dt
    ) const;

private:
    float sigma_to_t(float sigma) const;

    float time_shift(
        float mu,
        float sigma,
        float t
    ) const;

    int num_train_timesteps_;
    float shift_;
    bool use_dynamic_shifting_;
    std::optional<float> shift_terminal_;
    bool invert_sigmas_;
    std::string time_shift_type_;

    float sigma_max_;
    float sigma_min_;
};
