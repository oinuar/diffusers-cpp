#pragma once

#include "ggml/Tensor.hpp"

class Context;
class Graph;

class Compute {
public:
    virtual Tensor build(Context& ctx) = 0;
    virtual void compute(Graph& graph) = 0;
};
