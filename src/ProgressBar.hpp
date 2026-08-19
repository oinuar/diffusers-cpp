#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

class ProgressBar {
public:
    explicit ProgressBar(const std::string& title, int width = 20)
        : title_(title),
          total_(0),
          width_(width),
          lastTime_(Clock::now())
    {}

    ProgressBar() = default;

    void add(int value) {
        total_ += value;
    }

    void update(int currentIteration) {
        using namespace std::chrono;

        currentIteration = std::clamp(currentIteration, 0, total_);
        double progress = static_cast<double>(currentIteration) / total_;

        int filled = std::clamp(static_cast<int>(std::round(progress * width_)), 0, width_);

        auto now = Clock::now();
        double dt = duration<double>(now - lastTime_).count();

        double rate = 0.0;
        if (dt > 0.0)
            rate = (currentIteration - lastIteration_) / dt;

        std::cerr << '\r'
                  << title_ << ": |"
                  << std::string(filled, '#')
                  << std::string(width_ - filled, '-')
                  << "| "
                  << std::fixed << std::setprecision(1)
                  << currentIteration << '/' << total_;

        if (rate > 0.0) {
            if (rate >= 1.0)
                std::cerr << " [" << std::fixed << std::setprecision(2) << rate << " it/s]";
            else
                std::cerr << " [" << std::fixed << std::setprecision(2) << (1.0 / rate) << " s/it]";
        }

        if (currentIteration >= total_)
            std::cerr << " 100.0%" << std::endl;
        else
            std::cerr << ' ' << std::fixed << std::setprecision(1) << progress * 100.0 << '%';

        std::cerr << std::flush;

        lastIteration_ = currentIteration;
        lastTime_ = now;
    }

    void next() {
        update(lastIteration_ + 1);
    }

private:
    using Clock = std::chrono::steady_clock;

    std::string title_;
    int total_;
    int width_;

    int lastIteration_ = 0;
    Clock::time_point lastTime_;
};
