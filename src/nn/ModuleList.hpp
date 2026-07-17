#pragma once

#include "nn/Module.hpp"

class ModuleList : public Module {
public:
    explicit ModuleList(const std::initializer_list<std::shared_ptr<Module>>& values) : Module(), size_(values.size()) {
        size_t i = 0;

        for (auto it = std::begin(values); it != std::end(values); ++it)
            modules[std::to_string(i++)] = *it;
    }

    explicit ModuleList(const size_t& size) : Module(), size_(size) {
    }

    const std::shared_ptr<Module>& operator [](const size_t& index) const {
        return modules.at(std::to_string(index));
    }

    std::shared_ptr<Module>& operator [](const size_t& index) {
        return modules[std::to_string(index)];
    }

    const size_t& size() const {
        return size_;
    }

private:
    size_t size_;
};
