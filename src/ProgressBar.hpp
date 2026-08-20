#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

class ProgressBar {
public:
    explicit ProgressBar(const std::string& title,
                         int total_iterations = 0,
                         int width = 30)
        : width_(std::max(1, width))
    {
        root_ = std::make_unique<node>(
            make_state(title, total_iterations));

        current_ = root_.get();
    }

    ~ProgressBar()
    {
        if (rendered_lines_ > 0) {
            std::cerr << '\n';
        }
    }

    void update(int current_iteration)
    {
        update_state(
            current_->progress,
            current_iteration);

        render();
    }

    void next()
    {
        update(current_->progress.current_iteration + 1);
    }

    void push(const std::string& title,
              int total_iterations)
    {
        auto child = std::make_unique<node>(
            make_state(title, total_iterations));

        child->parent = current_;

        node* child_ptr = child.get();

        current_->children.push_back(
            std::move(child));

        current_ = child_ptr;

        render();
    }

    void pop()
    {
        // The root progress bar cannot be popped.
        if (current_ == root_.get()) {
            return;
        }

        // Complete the current progress bar.
        if (current_->progress.total_iterations > 0) {
            update_state(
                current_->progress,
                current_->progress.total_iterations);
        }

        /*
         * Keep the node in the tree.
         *
         * This is important: the node is now completed, but it
         * remains exactly where it was pushed.
         */
        current_ = current_->parent;

        render();
    }

private:
    using clock = std::chrono::steady_clock;

    struct progress_state {
        std::string title;

        int total_iterations = 0;
        int current_iteration = 0;

        int last_iteration = 0;

        double rate = 0.0;

        clock::time_point start_time;
        clock::time_point last_time;
    };

    struct node {
        explicit node(progress_state progress)
            : progress(std::move(progress))
        {
        }

        progress_state progress;

        node* parent = nullptr;

        std::vector<std::unique_ptr<node>> children;
    };

    static constexpr double rate_alpha_ = 0.15;

    int width_;

    std::unique_ptr<node> root_;
    node* current_ = nullptr;

    /*
     * Number of lines occupied by the previous rendering.
     *
     * The cursor is always left on the last rendered line.
     */
    std::size_t rendered_lines_ = 0;

    progress_state make_state(
        const std::string& title,
        int total_iterations) const
    {
        const auto now = clock::now();

        progress_state state;

        state.title = title;
        state.total_iterations =
            std::max(0, total_iterations);

        state.current_iteration = 0;
        state.last_iteration = 0;
        state.rate = 0.0;

        state.start_time = now;
        state.last_time = now;

        return state;
    }

    void update_state(
        progress_state& state,
        int current_iteration)
    {
        using namespace std::chrono;

        if (state.total_iterations > 0) {
            current_iteration =
                std::clamp(
                    current_iteration,
                    0,
                    state.total_iterations);
        } else {
            current_iteration =
                std::max(0, current_iteration);
        }

        const auto now = clock::now();

        const double dt =
            duration<double>(
                now - state.last_time).count();

        const int iteration_delta =
            current_iteration -
            state.last_iteration;

        if (dt > 0.0 && iteration_delta > 0) {
            const double instant_rate =
                static_cast<double>(
                    iteration_delta) / dt;

            if (state.rate <= 0.0) {
                state.rate = instant_rate;
            } else {
                state.rate =
                    (1.0 - rate_alpha_) * state.rate +
                    rate_alpha_ * instant_rate;
            }
        }

        state.current_iteration = current_iteration;
        state.last_iteration = current_iteration;
        state.last_time = now;
    }

    void render()
    {
        move_to_first_line();

        clear_previous_output();

        std::size_t line_count = 0;

        render_node(
            *root_,
            0,
            line_count);

        rendered_lines_ = line_count;

        std::cerr << std::flush;
    }

    void render_node(
        const node& current,
        std::size_t depth,
        std::size_t& line_count)
    {
        std::cerr
            << format_state(
                   current.progress,
                   depth,
                   &current == current_);

        ++line_count;

        for (const auto& child : current.children) {
            std::cerr << '\n';

            render_node(
                *child,
                depth + 1,
                line_count);
        }
    }

    void move_to_first_line()
    {
        if (rendered_lines_ <= 1) {
            if (rendered_lines_ == 1) {
                std::cerr << '\r';
            }

            return;
        }

        std::cerr
            << "\033["
            << rendered_lines_ - 1
            << 'A'
            << '\r';
    }

    void clear_previous_output()
    {
        if (rendered_lines_ == 0) {
            return;
        }

        /*
         * We are currently at the first line of the old output.
         */
        for (std::size_t i = 0;
             i < rendered_lines_;
             ++i) {

            std::cerr
                << "\033[2K"
                << '\r';

            if (i + 1 < rendered_lines_) {
                std::cerr << '\n';
            }
        }

        /*
         * Return to the first line.
         */
        if (rendered_lines_ > 1) {
            std::cerr
                << "\033["
                << rendered_lines_ - 1
                << 'A'
                << '\r';
        }

        rendered_lines_ = 0;
    }

    std::string format_state(
        const progress_state& state,
        std::size_t depth,
        bool active) const
    {
        using namespace std::chrono;

        const auto now = clock::now();

        const double elapsed =
            duration<double>(
                now - state.start_time).count();

        std::ostringstream output;

        /*
         * Indentation.
         *
         * Root:
         *
         * Generating: ...
         *
         * Child:
         *
         * └─ Denoising: ...
         *
         * Grandchild:
         *
         *    └─ Computing: ...
         */
        if (depth > 0) {
            output
                << std::string(
                       (depth - 1) * 3,
                       ' ')
                << "└─ ";
        }

        output
            << state.title
            << ": ";

        /*
         * Percentage and progress bar.
         */
        if (state.total_iterations > 0) {
            const double progress =
                static_cast<double>(
                    state.current_iteration) /
                state.total_iterations;

            const int percentage =
                std::clamp(
                    static_cast<int>(
                        std::floor(
                            progress * 100.0)),
                    0,
                    100);

            const int filled =
                std::clamp(
                    static_cast<int>(
                        std::round(
                            progress * width_)),
                    0,
                    width_);

            output
                << std::setw(3)
                << percentage
                << "%|"
                << std::string(
                       filled,
                       '#')
                << std::string(
                       width_ - filled,
                       ' ')
                << "| "
                << state.current_iteration
                << '/'
                << state.total_iterations;
        } else {
            output
                << state.current_iteration;
        }

        /*
         * Time information.
         */
        output
            << " ["
            << format_duration(elapsed);

        /*
         * ETA.
         */
        if (state.total_iterations > 0 &&
            state.current_iteration <
                state.total_iterations &&
            state.rate > 0.0) {

            const double remaining =
                static_cast<double>(
                    state.total_iterations -
                    state.current_iteration) /
                state.rate;

            output
                << '<'
                << format_duration(remaining);
        }

        /*
         * Rate.
         */
        output
            << ", "
            << format_rate(state.rate)
            << ']';

        return output.str();
    }

    static std::string format_rate(double rate)
    {
        if (rate <= 0.0) {
            return "?it/s";
        }

        std::ostringstream output;

        if (rate >= 1.0) {
            output
                << std::fixed
                << std::setprecision(2)
                << rate
                << "it/s";
        } else {
            output
                << std::fixed
                << std::setprecision(2)
                << (1.0 / rate)
                << "s/it";
        }

        return output.str();
    }

    static std::string format_duration(double seconds)
    {
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
            output
                << hours
                << ':'
                << std::setfill('0')
                << std::setw(2)
                << minutes
                << ':'
                << std::setw(2)
                << secs;
        } else {
            output
                << minutes
                << ':'
                << std::setfill('0')
                << std::setw(2)
                << secs;
        }

        return output.str();
    }
};