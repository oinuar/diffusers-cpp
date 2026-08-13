#include "diffusers/schedulers/FlowMatchEulerDiscreteScheduler.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

FlowMatchEulerDiscreteScheduler::FlowMatchEulerDiscreteScheduler(
    int num_train_timesteps,
    float shift,
    bool use_dynamic_shifting,
    std::optional<float> shift_terminal,
    bool invert_sigmas,
    std::string time_shift_type
)
    : num_train_timesteps_(num_train_timesteps),
      shift_(shift),
      use_dynamic_shifting_(use_dynamic_shifting),
      shift_terminal_(shift_terminal),
      invert_sigmas_(invert_sigmas),
      time_shift_type_(std::move(time_shift_type)),
      sigma_max_(1.0f),
      sigma_min_(1.0f / static_cast<float>(num_train_timesteps))
{
    if (num_train_timesteps_ <= 0) {
        throw std::invalid_argument(
            "FlowMatchEulerDiscreteScheduler: "
            "num_train_timesteps must be positive");
    }

    if (time_shift_type_ != "exponential" &&
        time_shift_type_ != "linear") {
        throw std::invalid_argument(
            "FlowMatchEulerDiscreteScheduler: "
            "unsupported time_shift_type: " +
            time_shift_type_);
    }
}

float FlowMatchEulerDiscreteScheduler::sigma_to_t(float sigma) const {
    return sigma * static_cast<float>(num_train_timesteps_);
}

float FlowMatchEulerDiscreteScheduler::time_shift(
    float mu,
    float sigma,
    float t
) const {
    if (t <= 0.0f || t >= 1.0f) {
        throw std::invalid_argument(
            "FlowMatchEulerDiscreteScheduler::time_shift(): "
            "t must be in (0, 1)");
    }

    const float exponent =
        std::pow(1.0f / t - 1.0f, sigma);

    if (time_shift_type_ == "exponential") {
        const float exp_mu = std::exp(mu);

        return exp_mu / (exp_mu + exponent);
    }

    // linear
    return mu / (mu + exponent);
}

Schedule FlowMatchEulerDiscreteScheduler::schedule(
    int num_inference_steps,
    float mu
) const {
    if (num_inference_steps <= 0) {
        throw std::invalid_argument(
            "FlowMatchEulerDiscreteScheduler::schedule(): "
            "num_inference_steps must be positive");
    }

    std::vector<float> sigmas;
    sigmas.reserve(num_inference_steps + 1);

    //
    // 1. Construct the base sigma schedule.
    //
    // Python:
    //
    //   sigmas = np.linspace(
    //       sigma_max,
    //       sigma_min,
    //       num_inference_steps
    //   )
    //
    const float t_max = sigma_to_t(sigma_max_);
    const float t_min = sigma_to_t(sigma_min_);

    if (num_inference_steps == 1) {
        sigmas.push_back(
            t_max /
            static_cast<float>(num_train_timesteps_)
        );
    } else {
        for (int i = 0; i < num_inference_steps; ++i) {
            const float t =
                t_max -
                static_cast<float>(i) *
                (t_max - t_min) /
                static_cast<float>(num_inference_steps - 1);

            sigmas.push_back(
                t /
                static_cast<float>(num_train_timesteps_)
            );
        }
    }

    //
    // 2. Apply FlowMatch timestep shifting.
    //
    if (use_dynamic_shifting_) {
        for (float& sigma : sigmas) {
            sigma = time_shift(
                mu,
                1.0f,
                sigma
            );
        }
    } else {
        for (float& sigma : sigmas) {
            sigma =
                shift_ * sigma /
                (
                    1.0f +
                    (shift_ - 1.0f) * sigma
                );
        }
    }

    //
    // 3. Stretch schedule to shift_terminal.
    //
    if (shift_terminal_.has_value()) {
        const float last = sigmas.back();

        const float scale_factor =
            (1.0f - last) /
            (1.0f - *shift_terminal_);

        for (float& sigma : sigmas) {
            sigma =
                1.0f -
                (1.0f - sigma) /
                scale_factor;
        }
    }

    //
    // 4. Convert sigma -> timestep.
    //
    std::vector<float> timesteps;
    timesteps.reserve(num_inference_steps);

    for (float sigma : sigmas) {
        timesteps.push_back(
            sigma_to_t(sigma)
        );
    }

    //
    // 5. Handle inverted sigmas.
    //
    if (invert_sigmas_) {
        for (float& sigma : sigmas) {
            sigma = 1.0f - sigma;
        }

        for (size_t i = 0; i < timesteps.size(); ++i) {
            timesteps[i] =
                sigma_to_t(sigmas[i]);
        }

        sigmas.push_back(1.0f);
    } else {
        //
        // Euler integration needs one additional sigma
        // representing the endpoint.
        //
        sigmas.push_back(0.0f);
    }

    return Schedule(
        std::move(timesteps),
        std::move(sigmas)
    );
}

Tensor FlowMatchEulerDiscreteScheduler::integrate(
    const Tensor& model_output,
    const Tensor& sample,
    const Tensor& dt
) const {
    //
    // Euler:
    //
    //   x_next = x + dt * model_output
    //
    return sample + dt * model_output;
}
