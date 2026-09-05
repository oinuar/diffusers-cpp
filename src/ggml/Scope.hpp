#pragma once

#include "ggml/ExecutionEngine.hpp"
#include <stdexcept>
#include <memory>

class Context;
class Engine;

class Scope {
public:
    Scope(Context& context, Engine& engine = ExecutionEngine::Default)
        : frame_(std::make_shared<Frame>(
              current_context_,
              current_engine_))
    {
        current_context_ = &context;
        current_engine_ = &engine;
    }

    // Set Engine in the scope, but does not change Context.
    explicit Scope(Engine& engine)
        : frame_(std::make_shared<Frame>(
              current_context_,
              current_engine_))
    {
        current_engine_ = &engine;
    }

    // Forks share the same scope frame.
    Scope(const Scope&) = default;

    Scope& operator=(const Scope&) = delete;
    Scope(Scope&&) = delete;
    Scope& operator=(Scope&&) = delete;

    ~Scope() {
        if (frame_.unique()) {
            current_context_ = frame_->previous_context;
            current_engine_ = frame_->previous_engine;
        }
    }

    static Context& context() {
        if (!current_context_)
            throw std::runtime_error("context(): No active Scope");

        return *current_context_;
    }

    static Engine& engine() {
        if (!current_engine_)
            throw std::runtime_error("engine(): No active Scope");

        return *current_engine_;
    }

private:
    struct Frame {
        Context* previous_context;
        Engine* previous_engine;

        Frame(Context* previous_context, Engine* previous_engine)
            : previous_context(previous_context)
            , previous_engine(previous_engine)
        {}
    };

    std::shared_ptr<Frame> frame_;

    inline static thread_local Context* current_context_ = nullptr;
    inline static thread_local Engine* current_engine_ = nullptr;
};
