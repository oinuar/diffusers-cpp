#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

class ProgressBar {
public:
    explicit ProgressBar(const std::string& title,
                         int total_iterations = 0,
                         int width = 30)
        : title_(title),
          total_(std::max(0, total_iterations)),
          width_(std::max(1, width)),
          start_time_(clock::now()),
          last_time_(start_time_)
    {}

    void update(int current_iteration) {
        using namespace std::chrono;

        if (total_ > 0) {
            current_iteration =
                std::clamp(current_iteration, 0, total_);
        } else {
            current_iteration =
                std::max(0, current_iteration);
        }

        const auto now = clock::now();

        const double dt =
            duration<double>(now - last_time_).count();

        const int iteration_delta =
            current_iteration - last_iteration_;

        if (dt > 0.0 && iteration_delta > 0) {
            const double instant_rate =
                static_cast<double>(iteration_delta) / dt;

            if (rate_ <= 0.0) {
                rate_ = instant_rate;
            } else {
                rate_ =
                    rate_ * (1.0 - rate_alpha_) +
                    instant_rate * rate_alpha_;
            }
        }

        const double elapsed =
            duration<double>(now - start_time_).count();

        double progress = 0.0;

        if (total_ > 0) {
            progress =
                static_cast<double>(current_iteration) /
                total_;
        }

        std::ostringstream output;

        output << '\r'
               << title_
               << ": ";

        // Percentage
        if (total_ > 0) {
            output << std::setw(3)
                   << static_cast<int>(
                          std::floor(progress * 100.0))
                   << "%|";
        } else {
            output << "    |";
        }

        // Progress bar
        int filled = 0;

        if (total_ > 0) {
            filled = std::clamp(
                static_cast<int>(
                    std::round(progress * width_)),
                0,
                width_);
        }

        output << std::string(filled, '#')
               << std::string(width_ - filled, ' ')
               << "| ";

        // Current / total
        if (total_ > 0) {
            output << current_iteration
                   << '/'
                   << total_;
        } else {
            output << current_iteration;
        }

        output << " ["
               << format_duration(elapsed);

        // ETA
        if (total_ > 0 &&
            current_iteration < total_ &&
            rate_ > 0.0) {

            const double remaining =
                static_cast<double>(
                    total_ - current_iteration) /
                rate_;

            output << '<'
                   << format_duration(remaining);
        } else {
            output << '<'
                   << format_duration(0.0);
        }

        output << ", ";

        // Rate
        if (rate_ > 0.0) {
            output << std::fixed
                << std::setprecision(2)
                << rate_
                << "it/s";
        } else {
            output << "?it/s";
        }

        output << ']';

        // Clear remnants of a previous, longer line
        output << "\033[K";

        std::cerr << output.str() << std::flush;

        last_iteration_ = current_iteration;
        last_time_ = now;

        if (total_ > 0 &&
            current_iteration >= total_) {
            std::cerr << '\n';
        }
    }

    void next() {
        update(last_iteration_ + 1);
    }

    void push(const std::string& title, int total_iterations) {
        // Save the current state
        state_.push_back({
            title_,
            total_,
            last_iteration_,
            rate_,
            start_time_
        });

        // Start the new nested progress state
        title_ = title;
        total_ = std::max(0, total_iterations);
        last_iteration_ = 0;
        rate_ = 0.0;
        start_time_ = clock::now();
        last_time_ = start_time_;

        update(0);
    }

    void pop() {
        if (state_.empty())
            return;

        const state previous = state_.back();
        state_.pop_back();

        // Restore the previous state
        title_ = previous.title;
        total_ = previous.total;
        last_iteration_ = previous.iteration;
        rate_ = previous.rate;
        start_time_ = previous.start_time;
        last_time_ = clock::now();

        // Redraw the restored progress bar
        update(last_iteration_);
    }

private:
    using clock = std::chrono::steady_clock;

    struct state {
        std::string title;
        int total;
        int iteration;
        double rate;
        clock::time_point start_time;
    };

    static constexpr double rate_alpha_ = 0.15;

    static std::string format_duration(double seconds) {
        seconds = std::max(0.0, seconds);

        const auto total_seconds =
            static_cast<long long>(seconds);

        const auto hours =
            total_seconds / 3600;

        const auto minutes =
            (total_seconds % 3600) / 60;

        const auto secs =
            total_seconds % 60;

        std::ostringstream output;

        if (hours > 0) {
            output << hours
                   << ':'
                   << std::setfill('0')
                   << std::setw(2)
                   << minutes
                   << ':'
                   << std::setw(2)
                   << secs;
        } else {
            output << minutes
                   << ':'
                   << std::setfill('0')
                   << std::setw(2)
                   << secs;
        }

        return output.str();
    }

    static std::string format_rate(double rate) {
        if (rate <= 0.0)
            return "?it/s";

        std::ostringstream output;
        output << std::fixed << std::setprecision(2);

        if (rate >= 1.0) {
            output << rate << "it/s";
        } else {
            output << (1.0 / rate) << "s/it";
        }

        return output.str();
    }

    std::string title_;

    int total_;
    int width_;

    int last_iteration_ = 0;

    double rate_ = 0.0;

    clock::time_point start_time_;
    clock::time_point last_time_;

    std::vector<state> state_;
};
