#include "diffusers/schedulers/FlowMatchEulerDiscreteScheduler.hpp"

FlowMatchEulerDiscreteScheduler::FlowMatchEulerDiscreteScheduler(
    int num_train_timesteps,
    float shift,
    bool use_dynamic_shifting,
    std::optional<float> shift_terminal,
    bool invert_sigmas,
    std::string time_shift_type
) : num_train_timesteps(num_train_timesteps),
    shift(shift),
    use_dynamic_shifting(use_dynamic_shifting),
    shift_terminal(shift_terminal),
    invert_sigmas(invert_sigmas),
    time_shift_type(time_shift_type),
    step_index(0)
{
    // Initialize default sigmas bounds matching Python's linspace logic
    sigma_max = 1.0f;
    sigma_min = 1.0f / static_cast<float>(num_train_timesteps);
}

float FlowMatchEulerDiscreteScheduler::sigma_to_t(float sigma) const {
    return sigma * num_train_timesteps;
}

float FlowMatchEulerDiscreteScheduler::time_shift(float mu, float sigma_val, float t_val) const {
    if (time_shift_type == "exponential") {
        float exp_mu = std::exp(mu);
        return exp_mu / (exp_mu + std::pow(1.0f / t_val - 1.0f, sigma_val));
    } else {
        // linear
        return mu / (mu + std::pow(1.0f / t_val - 1.0f, sigma_val));
    }
}

void FlowMatchEulerDiscreteScheduler::set_timesteps(
    int num_inference_steps,
    float mu,
    std::optional<std::vector<float>> sigmas_override
) {
    step_index = 0;
    std::vector<float> current_sigmas;

    if (sigmas_override.has_value()) {
        current_sigmas = sigmas_override.value();
        num_inference_steps = current_sigmas.size();
    } else {
        float t_max = sigma_to_t(sigma_max);
        float t_min = sigma_to_t(sigma_min);
        // Matches np.linspace(t_max, t_min, num_inference_steps)
        for (int i = 0; i < num_inference_steps; ++i) {
            float t_val = t_max - static_cast<float>(i) * (t_max - t_min) / static_cast<float>(num_inference_steps - 1);
            current_sigmas.push_back(t_val / num_train_timesteps);
        }
    }

    // 2. Perform timestep shifting
    if (use_dynamic_shifting) {
        for (float& s : current_sigmas) {
            s = time_shift(mu, 1.0f, s);
        }
    } else {
        for (float& s : current_sigmas) {
            s = shift * s / (1.0f + (shift - 1.0f) * s);
        }
    }

    // 3. Stretch shift to terminal if configured
    if (shift_terminal) {
        float last_val = current_sigmas.back();
        float scale_factor = (1.0f - last_val) / (1.0f - *shift_terminal);
        for (float& s : current_sigmas) {
            float one_minus_z = 1.0f - s;
            s = 1.0f - (one_minus_z / scale_factor);
        }
    }

    // Note: Karras, exponential, and beta sigma conversions are omitted here 
    // as they are typically disabled (False) in standard Flux configurations. 

    // 4 & 5. Convert to vectors and prepare timesteps
    sigmas.clear();
    timesteps.clear();
    for (float s : current_sigmas) {
        sigmas.push_back(s);
        timesteps.push_back(static_cast<int>(std::round(s * num_train_timesteps)));
    }

    // 6. Append terminal sigma value
    if (invert_sigmas) {
        for (float& s : sigmas) {
            s = 1.0f - s;
        }
        for (size_t i = 0; i < timesteps.size(); ++i) {
            timesteps[i] = static_cast<int>(std::round(sigmas[i] * num_train_timesteps));
        }
        sigmas.push_back(1.0f);
    } else {
        sigmas.push_back(0.0f);
    }

    reset_step_index();
}

const std::vector<int>& FlowMatchEulerDiscreteScheduler::get_timesteps() const {
    return timesteps;
}

void FlowMatchEulerDiscreteScheduler::reset_step_index() {
    step_index = 0;
}

float FlowMatchEulerDiscreteScheduler::step() {
    float sigma = sigmas[step_index];
    float sigma_next = sigmas[step_index + 1];
    float dt = sigma_next - sigma;
    
    ++step_index;

    return dt;
}

Tensor FlowMatchEulerDiscreteScheduler::integrate(
    const Tensor& model_output,
    const Tensor& sample,
    float dt
) {
    auto prev_sample = sample + dt * model_output;

    return prev_sample;
}
