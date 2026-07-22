#pragma once

#include "nn/Visitor.hpp"
#include "ggml/Tensor.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <unordered_map>
#include <string>
#include <fstream>
#include <filesystem>

class Backend;

class GGUFLoaderVisitor : public Visitor {
public:
    GGUFLoaderVisitor(Backend& backend, const std::filesystem::path& path);
    ~GGUFLoaderVisitor();

    virtual void visit(Parameter& parameter, std::vector<std::string> path);

private:
    ggml_context* ctx_;
    gguf_context* gguf_ctx_;
    ggml_backend_buffer_t buffer_;
    std::ifstream file_;
    std::unordered_map<std::string, size_t> lookup_;
};
