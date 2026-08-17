#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

class ProgressBar {
public:
    explicit ProgressBar(std::string title, int totalIterations, int width = 20)
        : title_(std::move(title)),
          total_(totalIterations),
          width_(width),
          lastTime_(Clock::now())
    {}

    void update(int currentIteration) {
        using namespace std::chrono;

        currentIteration = std::clamp(currentIteration, 0, total_);
        double progress = static_cast<double>(currentIteration) / total_;

        int filled = static_cast<int>(std::round(progress * width_));

        auto now = Clock::now();
        double dt = duration<double>(now - lastTime_).count();

        double rate = 0.0;
        if (dt > 0.0)
            rate = (currentIteration - lastIteration_) / dt;

        std::cerr << '\r'
                  << title_ << ": ["
                  << std::string(filled, '#')
                  << std::string(width_ - filled, '-')
                  << "] "
                  << std::fixed << std::setprecision(1)
                  << progress * 100.0 << "% ";

        if (rate > 0.0) {
            if (rate >= 1.0)
                std::cerr << std::setprecision(2) << rate << " it/s";
            else
                std::cerr << std::setprecision(2) << (1.0 / rate) << " s/it";
        }

        if (currentIteration == total_)
            std::cerr << " DONE" << std::endl;

        std::cerr << std::flush;

        lastIteration_ = currentIteration;
        lastTime_ = now;
    }

    void complete() {
        update(total_);
    }

private:
    using Clock = std::chrono::steady_clock;

    std::string title_;
    int total_;
    int width_;

    int lastIteration_ = 0;
    Clock::time_point lastTime_;
};
