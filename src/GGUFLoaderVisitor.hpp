#pragma once

#include "modules/Visitor.hpp"
#include "GGMLBackend.hpp"
#include "ggml.h"
#include "gguf.h"
#include <string>
#include <fstream>
#include <numeric>

#include <iostream>

class GGUFLoaderVisitor : public Visitor {
public:
    GGUFLoaderVisitor(GGMLBackend& backend, const std::string& path)
        : ctx_(nullptr), gguf_ctx_(nullptr), buffer_(nullptr), file_(path, std::ifstream::in | std::ifstream::binary), lookup_()
    {
        gguf_ctx_ = gguf_init_from_file(path.c_str(), {
            /*.no_alloc   =*/ true,
            /*.ctx        =*/ &ctx_,
        });

        auto n_tensors = gguf_get_n_tensors(gguf_ctx_);

        for (int i = 0; i < n_tensors; ++i) {
            auto name = gguf_get_tensor_name(gguf_ctx_, i);

            lookup_[name] = i;
        }

        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_, *backend);
        ggml_backend_buffer_set_usage(buffer_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }

    ~GGUFLoaderVisitor() {
        ggml_backend_buffer_free(buffer_);
        gguf_free(gguf_ctx_);
    }

    virtual void visit(Parameter<1>& parameter, std::vector<std::string> path) {
        visit_(parameter, std::move(path));
    }

    virtual void visit(Parameter<2>& parameter, std::vector<std::string> path) {
        visit_(parameter, std::move(path));
    }

    virtual void visit(Parameter<3>& parameter, std::vector<std::string> path) {
        visit_(parameter, std::move(path));
    }

    virtual void visit(Parameter<4>& parameter, std::vector<std::string> path) {
        visit_(parameter, std::move(path));
    }

private:
    ggml_context* ctx_;
    gguf_context* gguf_ctx_;
    ggml_backend_buffer_t buffer_;
    std::ifstream file_;
    std::unordered_map<std::string, size_t> lookup_;

    static std::string to_model_path(const std::vector<std::string>& path) {
        return std::accumulate(std::begin(path), std::end(path), std::string(""), [](const std::string& acc, const std::string& x) {
            if (acc.empty())
                return x;
            return acc + "." + x;
        });
    }

    template <size_t N>
    void visit_(Parameter<N>& parameter, std::vector<std::string> path) {
        auto model_path = to_model_path(path);
        auto it = lookup_.find(model_path);

        if (it == std::end(lookup_))
            throw std::runtime_error("Error while loading Tensor '" + model_path + "': not found");

        auto& tensor_id = it->second;
        auto tensor = ggml_get_tensor(ctx_, it->first.c_str());

        const std::streamoff offs = gguf_get_data_offset(gguf_ctx_) + gguf_get_tensor_offset(gguf_ctx_, tensor_id);
        std::vector<std::byte> buf(ggml_nbytes(tensor));

        std::cout << "Loading tensor " << it->first.c_str() << std::endl;

        file_.seekg(offs, file_.beg);
        
        if (!file_)
            throw std::runtime_error("Error while loading Tensor '" + model_path + "': seek failed");

        file_.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));

        if (file_.gcount() != static_cast<std::streamsize>(buf.size()))
            throw std::runtime_error("Error while loading Tensor '" + model_path + "': read failed");

        ggml_backend_tensor_set(tensor, buf.data(), 0, buf.size());

        parameter.set(Tensor(ctx_, tensor));
    }
};
