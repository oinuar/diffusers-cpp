#pragma once

#include "nn/Module.hpp"
#include "nn/Parameter.hpp"

class Linear : public Module {
public:
    Linear(int64_t in_features, int64_t out_features, bool bias = true);
    
    Tensor forward(Scope scope, Tensor x);

    std::shared_ptr<Parameter> weight() const;

    std::shared_ptr<Parameter> bias() const;

private:
    bool bias_;
};
