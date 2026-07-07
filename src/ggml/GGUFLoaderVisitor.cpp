#include "GGUFLoaderVisitor.hpp"
#include "Backend.hpp"
#include "nn/Parameter.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <string>
#include <fstream>
#include <numeric>

#include <iostream>

static std::string join_path(const std::vector<std::string>& path) {
    return std::accumulate(std::begin(path), std::end(path), std::string(""), [](const std::string& acc, const std::string& x) {
        if (acc.empty())
            return x;
        return acc + "." + x;
    });
}

GGUFLoaderVisitor::GGUFLoaderVisitor(Backend& backend, const std::string& path)
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

GGUFLoaderVisitor::~GGUFLoaderVisitor() {
    ggml_backend_buffer_free(buffer_);
    gguf_free(gguf_ctx_);
}

void GGUFLoaderVisitor::visit(Parameter& parameter, std::vector<std::string> path) {
    auto model_path = join_path(path);
    auto it = lookup_.find(model_path);

    if (it == std::end(lookup_))
        throw std::runtime_error("Error while loading Tensor '" + model_path + "': Tensor not found");

    auto& tensor_id = it->second;
    auto tensor = ggml_get_tensor(ctx_, it->first.c_str());

    const std::streamoff offs = gguf_get_data_offset(gguf_ctx_) + gguf_get_tensor_offset(gguf_ctx_, tensor_id);

    // Validate dimensions
    if (ggml_n_dims(tensor) != parameter.shape().rank())
        throw std::runtime_error("Error while loading Tensor '" + model_path + "': Parameter rank mismatch: expected " + std::to_string(parameter.shape().rank()) + ", got " + std::to_string(ggml_n_dims(tensor)));

    // Validate shape
    for (auto i = 0; i < ggml_n_dims(tensor); ++i) {
        if (tensor->ne[i] != parameter.shape()[i])
            throw std::runtime_error("Error while loading Tensor '" + model_path + "': Parameter shape mismatch: expected " + std::to_string(parameter.shape()[i]) + ", got " + std::to_string(tensor->ne[i]));
    }

    std::vector<std::byte> buf(ggml_nbytes(tensor));

    std::cout << "Loading tensor " << it->first.c_str() << " " << parameter.shape().to_string() << std::endl;

    file_.seekg(offs, file_.beg);
    
    if (!file_)
        throw std::runtime_error("Error while loading Tensor '" + model_path + "': seek failed");

    // Read the tensor data
    file_.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));

    if (file_.gcount() != static_cast<std::streamsize>(buf.size()))
        throw std::runtime_error("Error while loading Tensor '" + model_path + "': read failed");

    ggml_backend_tensor_set(tensor, buf.data(), 0, buf.size());

    parameter.set(Tensor(ctx_, tensor));
}
