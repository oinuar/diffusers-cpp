#pragma once

#include "modules/Module.hpp"
#include "modules/Visitor.hpp"
#include "Tensor.hpp"
#include <iostream>
#include <numeric>

class ParametersVisitor : public Visitor {
public:
    virtual void visit(Parameter<1>& parameter, std::vector<std::string> path) {
        std::cout << "path: " << to_dotted_path(path) << std::endl;
    }

    virtual void visit(Parameter<2>& parameter, std::vector<std::string> path) {
        std::cout << "path: " << to_dotted_path(path) << std::endl;
    }

private:
    static std::string to_dotted_path(const std::vector<std::string>& path) {
        return std::accumulate(std::begin(path), std::end(path), std::string(""), [](const std::string& acc, const std::string& x) {
            if (acc.empty())
                return x;
            return acc + "." + x;
        });
    }
};
