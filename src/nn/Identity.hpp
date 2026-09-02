#pragma once

#include "nn/Module.hpp"

class Identity : public Module {
public:
    Tensor forward(Scope, Tensor input) {
        return input;
    }
};
