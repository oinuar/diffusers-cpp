#pragma once

#include "ggml/Tensor.hpp"

#include <vector>

class Runtime;
class Graph;

class Compute {
public:
    class Plan {
    public:
        Plan(Tensor tensor) : tensors_({tensor}) {}
        Plan(std::vector<Tensor>&& tensors) : tensors_(tensors) {}

    private:
        std::vector<Tensor> tensors_;

        friend class Scheduler;
    };

    virtual Plan build(Runtime& runtime) = 0;
    virtual void compute(Graph& graph) = 0;
};
