#include "GGUFLoaderVisitor.hpp"
#include "Backend.hpp"
#include "nn/Parameter.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <string>
#include <fstream>
#include <numeric>

static std::string join_path(const std::vector<std::string>& path) {
    return std::accumulate(std::begin(path), std::end(path), std::string(""), [](const std::string& acc, const std::string& x) {
        if (acc.empty())
            return x;
        return acc + "." + x;
    });
}

static std::optional<std::filesystem::path> find_first_gguf(const std::filesystem::path& path)
{
    if (!std::filesystem::exists(path))
        return std::nullopt;

    if (!std::filesystem::is_directory(path))
        return path;

    for (const auto& entry : std::filesystem::directory_iterator(path))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".gguf")
            return entry.path();
    }

    return std::nullopt;
}

GGUFLoaderVisitor::GGUFLoaderVisitor(Backend& backend, const std::filesystem::path& path)
    : ctx_(nullptr), gguf_ctx_(nullptr), buffer_(nullptr), file_(), lookup_()
{
    auto gguf_path = find_first_gguf(path);

    if (!gguf_path)
        throw std::runtime_error("No such GGUF file in path: " + path.string());

    file_.open(*gguf_path, std::ifstream::in | std::ifstream::binary);

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

void GGUFLoaderVisitor::validate() const {
    std::string message;

    for (auto& [name, _] : lookup_) {
        if (!message.empty())
            message += "\n";

        message += "  - Tensor exists in checkpoint but was not loaded: " + name;
    }

    if (!message.empty())
        throw std::runtime_error("Error while validating the checkpoint:\n" + message);
}

void GGUFLoaderVisitor::visit(Parameter& parameter, std::vector<std::string> path) {
    auto model_path = join_path(path);
    auto it = lookup_.find(model_path);

    if (it == std::end(lookup_))
        throw std::runtime_error("Error while loading Tensor '" + model_path + "': Tensor not found");

    auto tensor_name = it->first;
    auto tensor_id = it->second;
    lookup_.erase(it);

    auto tensor = ggml_get_tensor(ctx_, tensor_name.c_str());

    const std::streamoff offs = gguf_get_data_offset(gguf_ctx_) + gguf_get_tensor_offset(gguf_ctx_, tensor_id);

    Tensor::Shape expected_shape(ggml_n_dims(tensor));

    for (auto r = 0; r < expected_shape.rank(); ++r)
        expected_shape[r] = tensor->ne[expected_shape.rank() - 1 - r];

    if (parameter.shape() != expected_shape)
        throw std::runtime_error("Error while loading Tensor '" + model_path + "': Parameter shape mismatch: expected " + parameter.shape().to_string() + ", got " + expected_shape.to_string());

    std::vector<std::byte> buf(ggml_nbytes(tensor));

    // std::cerr << "LOAD " << tensor_name.c_str() << " " << parameter.shape().to_string() << std::endl;

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
