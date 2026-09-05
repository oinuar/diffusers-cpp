#pragma once

#include "ggml/ExecutionEngine.hpp"
#include <stdexcept>
#include <memory>

class Context;
class Engine;

class Scope {
public:
    Scope(Context& context, Engine& engine = ExecutionEngine::Default)
        : previous_context_(current_context_)
        , previous_engine_(current_engine_)
    {
        current_context_ = &context;
        current_engine_ = &engine;
    }

    explicit Scope(Engine& engine)
        : previous_context_(current_context_)
        , previous_engine_(current_engine_)
    {
        current_context_ = nullptr;
        current_engine_ = &engine;
    }

    Scope(const Scope& other)
        : previous_context_(current_context_)
        , previous_engine_(current_engine_)
    {
        current_context_ = other.current_context_;
        current_engine_ = other.current_engine_;
    }

    ~Scope() {
        current_context_ = previous_context_;
        current_engine_ = previous_engine_;
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
    inline static thread_local Context* current_context_ = nullptr;
    inline static thread_local Engine* current_engine_ = nullptr;

    Context* previous_context_;
    Engine* previous_engine_;
};
