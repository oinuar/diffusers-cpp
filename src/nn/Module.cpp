#include "nn/Module.hpp"
#include "nn/Visitor.hpp"

void Module::accept(Visitor& visitor, std::vector<std::string> path) {
    for (auto& [name, child] : modules) {
        auto child_path = path;

        child_path.push_back(name);

        child->accept(visitor, std::move(child_path));
    }

    visitor.visit(*this, std::move(path));
}
