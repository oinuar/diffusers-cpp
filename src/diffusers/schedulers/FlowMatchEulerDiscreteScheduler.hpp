#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <optional>
#include <limits>
#include "ggml/Tensor.hpp"

class Runtime;

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

    void set_timesteps(
        Runtime& runtime,
        int num_inference_steps,
        float mu = 0.0f,
        std::optional<std::vector<float>> sigmas_override = std::nullopt
    );

    std::vector<int> get_timesteps() const;

    std::vector<Tensor> step(
        Runtime& runtime,
        const Tensor& model_output,
        int timestep,
        const Tensor& sample,
        bool return_dict = false
    );

    void reset_step_index();

private:
    int num_train_timesteps;
    float shift;
    bool use_dynamic_shifting;
    std::optional<float> shift_terminal;
    bool invert_sigmas;
    std::string time_shift_type;

    std::vector<float> sigmas;
    std::vector<int> timesteps;
    int step_index;

    float sigma_max;
    float sigma_min;

    float sigma_to_t(float sigma) const;
    float time_shift(float mu, float sigma_val, float t_val) const;
    void init_step_index(int timestep);
};
