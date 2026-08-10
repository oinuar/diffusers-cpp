#pragma once

#include <vector>
#include <string>
#include <cstddef>

class Parameter;
class Module;

class Visitor {
public:
    virtual void visit(Parameter&, std::vector<std::string>) {}
    virtual void visit(Module&, std::vector<std::string>) {}
};
