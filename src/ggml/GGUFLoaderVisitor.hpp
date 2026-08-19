#pragma once

#include "nn/Visitor.hpp"
#include <unordered_map>
#include <string>
#include <fstream>
#include <filesystem>
#include <ggml.h>
#include <gguf.h>

class Runtime;

class GGUFLoaderVisitor : public Visitor {
public:
    GGUFLoaderVisitor(Runtime& runtime, const std::filesystem::path& path);
    ~GGUFLoaderVisitor();

    virtual void visit(Parameter& parameter, std::vector<std::string> path);

    void validate() const;

    size_t size() const {
        return lookup_.size();
    }

private:
    Runtime& runtime_;
    gguf_context* gguf_ctx_;
    std::shared_ptr<std::ifstream> file_;
    std::unordered_map<std::string, size_t> lookup_;
};
