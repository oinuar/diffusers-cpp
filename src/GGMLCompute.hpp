#pragma once

#include "Tensor.hpp"

#include <unordered_map>

class GGMLContext;
class GGMLGraph;

class GGMLCompute {
public:
    typedef std::unordered_map<std::string, Tensor> Parameters;

    virtual std::pair<Parameters, Tensor> build(GGMLContext& ctx) = 0;

    virtual void compute(GGMLGraph& graph) = 0;
};