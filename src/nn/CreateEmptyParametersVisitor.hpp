#pragma once

#include "nn/Visitor.hpp"
#include "nn/Parameter.hpp"
#include "nn/ModulePath.hpp"

class CreateEmptyParametersVisitor : public Visitor {
public:
    explicit CreateEmptyParametersVisitor(const std::string& prefix = "")
        : prefix_(prefix)
    {}

    void visit(Parameter& parameter, std::vector<std::string> path) override {
        auto tensor = Tensor::empty<float>(parameter.shape());

        ModulePath module_path;
        tensor.name(module_path(path).c_str());

        parameter.set(tensor);
    }

private:
    std::string prefix_;
};
