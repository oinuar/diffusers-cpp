#pragma once

#include "ggml/Context.hpp"
#include "ggml/Computation.hpp"
#include "ggml/Backend.hpp"
#include "ggml/MetaDevice.hpp"
#include "ggml/Scheduler.hpp"
#include "ggml/Allocator.hpp"
#include "ggml/ShardingAllocator.hpp"
#include "ggml/ExecutionRuntime.hpp"
#include "nn/Visitor.hpp"
#include "nn/Parameter.hpp"
#include "nn/ModulePath.hpp"
#include "./ArgumentParser.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

class TestCLI {
public:
    int main() {
        ggml_time_init();
        ggml_log_set([](ggml_log_level, const char* text, void*) { std::cerr << text; }, nullptr);

        ggml_backend_load_all();

        // This controls how many fake devices are used to run the tests.
        auto n_devices = args_.get_optional<size_t>("--runner-n_devices").value_or(1);
        auto use_gpu = args_.get_optional<bool>("--runner-use_gpu").value_or(false);

        Context weights_context(get_graph_size());

        auto computation = compute(weights_context);

        // If more than one device, use Meta device.
        if (n_devices > 1) {
            if (use_gpu)
                throw std::runtime_error("Multi-GPU tests are not supported");

            Device cpu(GGML_BACKEND_DEVICE_TYPE_CPU);
            std::vector<ggml_backend_dev_t> devices;

            for (auto i = 0; i < n_devices; ++i)
                devices.push_back(*cpu);

            MetaDevice meta(std::move(devices));
            Backend meta_backend(meta);
            Backend cpu_backend(cpu);
            Scheduler scheduler({&meta_backend, &cpu_backend}, get_graph_size());
            Allocator weights_allocator(meta, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            Allocator state_allocator(meta, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

            return run(scheduler, weights_allocator, state_allocator, computation);
        }

        if (use_gpu) {
            Device cpu(GGML_BACKEND_DEVICE_TYPE_CPU);
            Device gpu(GGML_BACKEND_DEVICE_TYPE_GPU);
            Backend cpu_backend(cpu);
            Backend gpu_backend(gpu);
            Scheduler scheduler({&gpu_backend, &cpu_backend}, get_graph_size());
            Allocator weights_allocator(gpu, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            Allocator state_allocator(gpu, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

            return run(scheduler, weights_allocator, state_allocator, computation);
        }

        Device cpu(GGML_BACKEND_DEVICE_TYPE_CPU);
        Backend cpu_backend(cpu);
        Scheduler scheduler({&cpu_backend}, get_graph_size());
        Allocator weights_allocator(cpu, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        Allocator state_allocator(cpu, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

        return run(scheduler, weights_allocator, state_allocator, computation);
    }

    virtual Computation<std::vector<Tensor>> compute(Context& weights_context) = 0;

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
            throw std::runtime_error("tensor data size does not match shape, expected " + std::to_string(expected) + ", but got " + std::to_string(data.size()));

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
        CreateParametersVisitor(Context& context, const ArgumentParser& args, const std::string& prefix = "")
            : context_(context), args_(args), prefix_(prefix)
        {}

        virtual void visit(Parameter& parameter, std::vector<std::string> path) {
            ModulePath module_path("-", "--param");
            auto joined_path = module_path(path, prefix_);

            ArgumentParser::parser<Tensor> parser(context_, parameter.dtype());
            
            auto tensor = parser(joined_path, get_param(joined_path));

            parameter.set(tensor, joined_path);
        }

        std::string get_param(const std::string& path) {
            auto value = args_.get_one<std::string>(path);

            // Read tensor value from file
            std::error_code ec;
            if (std::filesystem::is_regular_file(value, ec)) {
                std::ifstream file(value);
                if (!file)
                    throw std::runtime_error(
                        "Failed to open parameter file: " + value);
                
                std::stringstream buffer;
                buffer << file.rdbuf();

                return buffer.str();
            }

            // Otherwise, read inline tensor
            return value;
        }

        Context& context() {
            return context_;
        }

        const std::string& prefix() const {
            return prefix_;
        }

    private:
        Context& context_;
        const ArgumentParser& args_;
        std::string prefix_;
    };
private:
    int run(Scheduler& scheduler, Allocator& weights_allocator, Allocator& state_allocator, Computation<std::vector<Tensor>> computation) {
        std::mt19937 rng;
        auto results = ExecutionRuntime::Default.run(scheduler, weights_allocator, state_allocator, rng, computation);

        for (auto& tensor : results) {
            switch (tensor.dtype()) {
            case Tensor::DType<float>::value:
            {
                auto data = ExecutionRuntime::Default.read<float>(tensor);
                print_tensor_like(data, tensor.shape());
                break;
            }

            case Tensor::DType<int8_t>::value:
            {
                auto data = ExecutionRuntime::Default.read<int8_t>(tensor);
                print_tensor_like(data, tensor.shape());
                break;
            }

            case Tensor::DType<int16_t>::value:
            {
                auto data = ExecutionRuntime::Default.read<int16_t>(tensor);
                print_tensor_like(data, tensor.shape());
                break;
            }

            case Tensor::DType<int32_t>::value:
            {
                auto data = ExecutionRuntime::Default.read<int32_t>(tensor);
                print_tensor_like(data, tensor.shape());
                break;
            }

            default:
                throw std::runtime_error("Unsupported tensor type: " + std::string(ggml_type_name(tensor.dtype())));
            }
        }

        return EXIT_SUCCESS;
    }

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
