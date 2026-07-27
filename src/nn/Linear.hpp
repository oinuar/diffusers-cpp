#pragma once

#include "nn/Module.hpp"

class Linear : public Module {
public:
    Linear(int64_t in_features, int64_t out_features, bool bias = true);
    
    Tensor forward(Runtime& runtime, Tensor x);

private:
    bool bias_;
};
