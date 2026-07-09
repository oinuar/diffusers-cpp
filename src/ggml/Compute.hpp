#pragma once

#include "ggml/Tensor.hpp"

#include <vector>

class Context;
class Graph;

class Compute {
public:
    typedef std::vector<Tensor> Plan;

    virtual Plan build(Context& ctx) = 0;
    virtual void compute(Graph& graph) = 0;
};
