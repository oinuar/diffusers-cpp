#pragma once

#include <unordered_map>
#include <string>
#include <memory>
#include "Tensor.hpp"
#include "modules/Visitor.hpp"

class Module {
public:
    typedef std::unordered_map<std::string, std::shared_ptr<Module>> Children;

    virtual void accept(Visitor& visitor, std::vector<std::string> path = std::vector<std::string>()) {
        for (auto& [name, child] : modules) {
            auto child_path = path;

            child_path.push_back(name);

            child->accept(visitor, std::move(child_path));
        }

        visitor.visit(*this, std::move(path));
    }

protected:
    Children modules;
};
