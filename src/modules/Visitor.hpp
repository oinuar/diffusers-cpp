#pragma once

#include <vector>

template <size_t N> class Parameter;
class Module;

class Visitor {
public:
    virtual void visit(Parameter<1>&, std::vector<std::string>) {}
    virtual void visit(Parameter<2>&, std::vector<std::string>) {}
    virtual void visit(Parameter<3>&, std::vector<std::string>) {}
    virtual void visit(Parameter<4>&, std::vector<std::string>) {}

    virtual void visit(Module&, std::vector<std::string>) {}
};
