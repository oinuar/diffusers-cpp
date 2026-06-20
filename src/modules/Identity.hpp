#pragma once

#include "modules/Module.hpp"

class Identity : public Module {
public:
    Tensor forward(ggml_context*, Tensor input) {
        return input;
    }
};
