#pragma once

#include "Tensor.hpp"

class GGMLContext;
class GGMLGraph;

class GGMLCompute {
public:
    virtual Tensor build(GGMLContext& ctx) = 0;
    virtual void compute(GGMLGraph& graph) = 0;
};
