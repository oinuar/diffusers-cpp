#pragma once

#include "ggml/Tensor.hpp"

#include <vector>

class Context;
class Graph;

class Compute {
public:
    class Plan {
    public:
        Plan(Tensor tensor) : tensors_({tensor}) {}
        Plan(std::vector<Tensor>&& tensors) : tensors_(tensors) {}

        std::vector<Tensor>& tensors() { return tensors_; }

    private:
        std::vector<Tensor> tensors_;
    };

    virtual Plan build(Context& ctx) = 0;
    virtual void compute(Graph& graph) = 0;
};
