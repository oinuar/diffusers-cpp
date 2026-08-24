#pragma once

#include "ggml/Graph.hpp"
#include "ggml/Computation.hpp"
#include "ggml/Context.hpp"
#include "ggml/Backend.hpp"
#include "ggml/MetaDevice.hpp"
#include "ggml/Scheduler.hpp"
#include "nn/Visitor.hpp"
#include "nn/Parameter.hpp"
#include "./ArgumentParser.hpp"
#include <iostream>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <vector>

class TestCLI {
public:
    int main() {
        ggml_time_init();
        ggml_log_set([](ggml_log_level, const char* text, void*) { std::cerr << text; }, nullptr);

        ggml_backend_load_all();
        // With --n-devices N, run on a meta device sharded across N CPU
        // backends (tensor parallelism); the plain CPU backend stays last,
        // as the ggml scheduler requires.
        std::unique_ptr<MetaDevice> meta_dev;
        std::vector<Backend> backends; // wrappers keep the ggml backends alive

        if (auto n_devices = args_.get_optional<int>("--n-devices")) {
            const auto cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            if (cpu_dev == nullptr)
                throw std::runtime_error("no CPU device found for --n-devices");

            meta_dev = std::make_unique<MetaDevice>(std::vector<ggml_backend_dev_t>(*n_devices, cpu_dev));
            backends.emplace_back(**meta_dev);
            backends.emplace_back(GGML_BACKEND_DEVICE_TYPE_CPU);
        } else {
            backends.emplace_back(GGML_BACKEND_DEVICE_TYPE_CPU);
        }

        std::vector<ggml_backend_t> backend_pointers;
        for (auto& backend : backends)
            backend_pointers.push_back(*backend);

        Scheduler scheduler(std::move(backend_pointers), get_graph_size());
        Context context(scheduler.capacity());
        Runtime runtime(scheduler, context, std::random_device{}(), meta_dev.get());

        auto results = compute(runtime);

        for (auto& tensor : results) {
            switch (tensor.dtype()) {
            case Tensor::DType<float>::value:
            {
                auto data = runtime.value<float>(tensor);
                print_tensor_like(data, tensor.shape());
                break;
            }

            case Tensor::DType<int8_t>::value:
            {
                auto data = runtime.value<int8_t>(tensor);
                print_tensor_like(data, tensor.shape());
                break;
            }

            case Tensor::DType<int16_t>::value:
            {
                auto data = runtime.value<int16_t>(tensor);
                print_tensor_like(data, tensor.shape());
                break;
            }

            case Tensor::DType<int32_t>::value:
            {
                auto data = runtime.value<int32_t>(tensor);
                print_tensor_like(data, tensor.shape());
                break;
            }

            default:
                throw std::runtime_error("Unsupported tensor type: " + std::string(ggml_type_name(tensor.dtype())));
            }
        }

        return EXIT_SUCCESS;
    }

    virtual std::vector<Tensor> compute(Runtime& runtime) = 0;

    virtual size_t get_graph_size() const {
        return GGML_DEFAULT_GRAPH_SIZE;
    }

    const ArgumentParser& args() const {
        return args_;
    }

    template <typename T>
    void print_tensor_like(const std::vector<T>& data, const Tensor::Shape& shape, std::ostream& stream=std::cout) const {
        size_t expected = 1;
        for (auto i = 0; i < shape.rank(); ++i)
            expected *= shape[i];

        if (expected != data.size())
            throw std::runtime_error("tensor data size does not match shape");

        std::cerr << "output shape: " << shape.to_string() << std::endl;

        size_t index = 0;
        print_tensor_recursively(data, shape, 0, index, stream);
        stream << std::endl;
    }

protected:
    ArgumentParser args_;

    TestCLI(int argc, char** argv) : args_(argc, argv) {}

public:
    class CreateParametersVisitor : public Visitor {
    public:
        CreateParametersVisitor(Runtime& runtime, const ArgumentParser& args, const std::string& prefix = "")
            : runtime_(runtime), args_(args), prefix_(prefix)
        {}

        virtual void visit(Parameter& parameter, std::vector<std::string> path) {
            auto joined_path = join_path(path, prefix_);

            auto tensor_value = args_.get_one<std::string>(joined_path);
            ArgumentParser::parser<Tensor> parser(runtime_, &parameter);
            Tensor tensor;

            // Read tensor value from file
            std::error_code ec;
            if (std::filesystem::is_regular_file(tensor_value, ec)) {
                std::ifstream file(tensor_value);
                if (!file)
                    throw std::runtime_error(
                        "Failed to open parameter file: " + tensor_value);
                
                std::stringstream buffer;
                buffer << file.rdbuf();

                tensor = parser(joined_path, buffer.str());
            }

            // Otherwise, read inline tensor
            else
                tensor = parser(joined_path, tensor_value);
            
            parameter.set(tensor);
        }

    private:
        Runtime& runtime_;
        const ArgumentParser& args_;
        std::string prefix_;
        
        static std::string join_path(const std::vector<std::string>& path, const std::string& prefix = "") {
            std::string seed("--param");

            if (!prefix.empty()) {
                seed += '-';
                seed += prefix;
            }

            return std::accumulate(std::begin(path), std::end(path), seed, [](const std::string& acc, const std::string& x) {
                return acc + "-" + x;
            });
        }
    };
private:
    void print_escaped_string(const std::string& value, std::ostream& stream) const {
        stream << '"';

        for (char c : value) {
            switch (c) {
            case '"':
                stream << "\\\"";
                break;
            case '\\':
                stream << "\\\\";
                break;
            case '\n':
                stream << "\\n";
                break;
            case '\r':
                stream << "\\r";
                break;
            case '\t':
                stream << "\\t";
                break;
            default:
                stream << c;
                break;
            }
        }

        stream << '"';
    }

    template <typename T>
    void print_tensor_recursively(const std::vector<T>& data, const Tensor::Shape& shape, size_t dim, size_t& index, std::ostream& stream) const {
        if (shape.rank() == 0 && dim == 0) {
            stream << data[index++];
            return;
        }

        stream << "[";

        if (dim == shape.rank() - 1)
        {
            // Last dimension: print elements
            for (auto i = 0; i < shape[dim]; ++i)
            {
                if constexpr (std::is_same_v<T, std::string>)
                    print_escaped_string(data[index++], stream);
                else
                    stream << data[index++];

                if (i + 1 != shape[dim])
                    stream << ", ";
            }
        }
        else
        {
            // Print nested arrays
            for (auto i = 0; i < shape[dim]; ++i)
            {
                print_tensor_recursively(data, shape, dim + 1, index, stream);
                if (i + 1 != shape[dim])
                    stream << ", ";
            }
        }

        stream << "]";
    }
};
