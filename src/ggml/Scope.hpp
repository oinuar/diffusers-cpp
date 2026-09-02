#pragma once

#include <stdexcept>

class Context;

class Scope {
public:
    Scope(Context& context) : previous_(current_) {
        current_ = &context;
    }

    Scope(const Scope& other) : previous_(current_) {
        current_ = other.current_;
    }

    ~Scope() {
        current_ = previous_;
    }

    static Context& context() {
        if (current_ == nullptr)
            throw std::runtime_error("No active Scope, did you forget to create it?");

        return *current_;
    }

private:
    inline static thread_local Context* current_ = nullptr;
    Context* previous_;
};
