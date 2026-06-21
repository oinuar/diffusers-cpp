#pragma once

#include "Tensor.hpp"

class GGMLContext;
class GGMLGraph;

class GGMLCompute {
public:
    typedef std::vector<Tensor> Dependencies;

    virtual std::pair<Dependencies, Tensor> build(GGMLContext& ctx) = 0;

    virtual void compute(GGMLGraph& graph) = 0;
};