#pragma once

#include <unordered_map>
#include <string>
#include <memory>
#include "Tensor.hpp"

class Visitor;

class Module {
public:
    typedef std::unordered_map<std::string, std::shared_ptr<Module>> Children;

    Module() = default;

    virtual void accept(Visitor& visitor, std::vector<std::string> path = std::vector<std::string>());

protected:
    Children modules;
};
