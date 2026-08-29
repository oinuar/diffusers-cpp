#include "ggml/GGUFLoaderVisitor.hpp"
#include "ggml/Runtime.hpp"
#include "ggml/Allocator.hpp"
#include "nn/Parameter.hpp"
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

GGUFLoaderVisitor::GGUFLoaderVisitor(Runtime& runtime, const std::filesystem::path& path, Allocator* allocator)
    : runtime_(runtime), gguf_ctx_(nullptr), file_(std::make_shared<std::ifstream>()), lookup_(), allocator_(allocator)
{
    auto gguf_path = find_first_gguf(path);

    if (!gguf_path)
        throw std::runtime_error(
            "No such GGUF file in path: " + path.string());

    file_->open(*gguf_path, std::ifstream::in | std::ifstream::binary);

    if (!file_->is_open())
        throw std::runtime_error("Failed to open GGUF file: " + gguf_path->string());

    //
    // Parse GGUF metadata only.
    //
    // No ggml context is requested here. Therefore
    // gguf_ctx_ does not own any ggml tensors.
    //
    gguf_ctx_ = gguf_init_from_file(
        gguf_path->c_str(),
        {
            /* .no_alloc = */ true,
            /* .ctx      = */ nullptr,
        });

    if (!gguf_ctx_)
        throw std::runtime_error(
            "Failed to initialize GGUF file: " +
            gguf_path->string());

    auto n_tensors = gguf_get_n_tensors(gguf_ctx_);
    lookup_.reserve(n_tensors);

    for (auto i = 0; i < n_tensors; ++i) {
        auto name = gguf_get_tensor_name(gguf_ctx_, i);

        if (!name)
            throw std::runtime_error(
                "GGUF tensor has no name");

        lookup_[name] = i;
    }
}

GGUFLoaderVisitor::~GGUFLoaderVisitor() {
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

    auto type = gguf_get_tensor_type(gguf_ctx_, tensor_id);
    auto ne = gguf_get_tensor_ne(gguf_ctx_, tensor_id);
    int n_dims = 0;

    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (ne[i] > 1 || (i == 0 && ne[0] == 1)) {
            n_dims = i + 1;
        }
    }

    if (n_dims == 0)
        throw std::runtime_error(
            "GGUF tensor has invalid dimensions");

    auto name = gguf_get_tensor_name(gguf_ctx_, tensor_id);

    const std::streamoff offs = gguf_get_data_offset(gguf_ctx_) + gguf_get_tensor_offset(gguf_ctx_, tensor_id);

    Tensor::Shape expected_shape(n_dims);

    for (auto r = 0; r < expected_shape.rank(); ++r)
        expected_shape[r] = ne[expected_shape.rank() - 1 - r];

    if (parameter.shape() != expected_shape)
        throw std::runtime_error("Error while loading Tensor '" + model_path + "': Parameter shape mismatch: expected " + parameter.shape().to_string() + ", got " + expected_shape.to_string());

    auto tensor = Tensor::empty(*runtime_.context(), expected_shape, type, allocator_);

    ggml_set_name(*tensor, name);

    auto n_bytes = ggml_nbytes(*tensor);

    runtime_.bind<std::byte>(tensor,
        [n_bytes, expected_shape, offs, file = file_, model_path = std::move(model_path)/*, tensor_name = std::move(tensor_name)*/](std::mt19937&) {
            std::vector<std::byte> buf(n_bytes);

            //std::cerr << "LOAD " << tensor_name.c_str() << " " << expected_shape.to_string() << std::endl;

            file->seekg(offs, file->beg);
            
            if (!*file)
                throw std::runtime_error("Error while loading Tensor '" + model_path + "': seek failed");

            // Read the tensor data
            file->read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));

            if (file->gcount() != static_cast<std::streamsize>(buf.size()))
                throw std::runtime_error("Error while loading Tensor '" + model_path + "': read failed");

            return buf;
        }, /*once=*/true);

    parameter.set(tensor);

    // TODO: This is separate responsibility because should work for all tensors, not just parameters
    for (auto& backend : runtime_.scheduler().backends())
        backend->device().visit(parameter, path);
}
