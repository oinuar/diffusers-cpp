#pragma once

#include "nn/Module.hpp"

class Identity : public Module {
public:
    Tensor forward(Context&, Tensor input) {
        return input;
    }
};
