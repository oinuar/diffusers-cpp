#pragma once

#include "nn/Module.hpp"

class Identity : public Module {
public:
    Tensor forward(Runtime&, Tensor input) {
        return input;
    }
};
