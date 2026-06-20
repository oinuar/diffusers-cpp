#pragma once

#include <unordered_map>
#include <string>
#include <memory>
#include <functional>
#include "Tensor.hpp"
#include "Parameter.hpp"

class Module {
public:
    typedef std::unordered_map<std::string, Parameter> Parameters;
    typedef std::unordered_map<std::string, std::shared_ptr<Module>> Modules;

protected:
    Parameters params;
    Modules modules;
};
