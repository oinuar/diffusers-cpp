#pragma once

#include <cstddef>
#include <vector>

#include "ggml/Tensor.hpp"

class Schedule {
public:
    struct Step {
        float timestep;
        float sigma;
        float sigma_next;
        float dt;
    };

    class iterator {
    public:
        using value_type = Step;

        iterator(const Schedule& schedule, size_t index)
            : schedule_(schedule), index_(index) {}

        Step operator*() const {
            return schedule_[index_];
        }

        iterator& operator++() {
            ++index_;
            return *this;
        }

        bool operator==(const iterator& other) const {
            return index_ == other.index_ && &schedule_ == &other.schedule_;
        }

        bool operator!=(const iterator& other) const {
            return !(*this == other);
        }

    private:
        const Schedule& schedule_;
        size_t index_;
    };

    Schedule() = default;

    Schedule(
        std::vector<float> timesteps,
        std::vector<float> sigmas
    )
        : timesteps_(std::move(timesteps)),
          sigmas_(std::move(sigmas))
    {
        if (sigmas_.size() != timesteps_.size() + 1) {
            throw std::invalid_argument(
                "Schedule: expected sigmas.size() == "
                "timesteps.size() + 1");
        }
    }

    size_t size() const {
        return timesteps_.size();
    }

    Step operator[](size_t index) const {
        if (index >= size()) {
            throw std::out_of_range(
                "Schedule: step index out of range");
        }

        const float sigma = sigmas_[index];
        const float sigma_next = sigmas_[index + 1];

        return Step{
            /*.timestep = */ timesteps_[index],
            /* .sigma = */ sigma,
            /* .sigma_next = */ sigma_next,
            /* .dt = */ sigma_next - sigma,
        };
    }

    iterator begin() const {
        return iterator(*this, 0);
    }

    iterator end() const {
        return iterator(*this, size());
    }

    const std::vector<float>& timesteps() const {
        return timesteps_;
    }

    const std::vector<float>& sigmas() const {
        return sigmas_;
    }

private:
    std::vector<float> timesteps_;
    std::vector<float> sigmas_;
};
