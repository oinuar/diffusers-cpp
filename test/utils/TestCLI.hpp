#pragma once

#include "ggml/Graph.hpp"
#include "ggml/Computation.hpp"
#include "ggml/Context.hpp"
#include "ggml/Backend.hpp"
#include "ggml/MetaDevice.hpp"
#include "ggml/Scheduler.hpp"
#include "ggml/DeviceAllocator.hpp"
#include "ggml/SchedulerAllocator.hpp"
#include "nn/Visitor.hpp"
#include "nn/Parameter.hpp"
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

        Context context(get_graph_size());

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

            DeviceAllocator allocator(context, meta);

            return main(scheduler, context, allocator, meta);
        }

        if (use_gpu) {
            Device cpu(GGML_BACKEND_DEVICE_TYPE_CPU);
            Device gpu(GGML_BACKEND_DEVICE_TYPE_GPU);
            Backend cpu_backend(cpu);
            Backend gpu_backend(gpu);
            Scheduler scheduler({&gpu_backend, &cpu_backend}, get_graph_size());

            DeviceAllocator allocator(context, gpu);

            return main(scheduler, context, allocator, gpu);
        }

        Device cpu(GGML_BACKEND_DEVICE_TYPE_CPU);
        Backend cpu_backend(cpu);
        Scheduler scheduler({&cpu_backend}, get_graph_size());

        SchedulerAllocator allocator;

        return main(scheduler, context, allocator, cpu);
    }

    virtual std::vector<Tensor> compute(Scheduler& scheduler, Context& context, Allocator& allocator, std::optional<Context>& local_context, std::optional<DeviceAllocator>& local_allocator) = 0;

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
            auto joined_path = join_path(path, prefix_);

            auto tensor_value = args_.get_one<std::string>(joined_path);
            ArgumentParser::parser<Tensor> parser(context_);
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
        Context& context_;
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
    int main(Scheduler& scheduler, Context& context, Allocator& allocator, const Device& device) {
        auto use_local_context = args_.get_optional<bool>("--runner-use_local_context").value_or(false);

        std::optional<Context> local_context;
        std::optional<DeviceAllocator> local_allocator;

        if (use_local_context) {
            local_context.emplace(get_graph_size());
            local_allocator.emplace(*local_context, device);
        }

        auto results = compute(scheduler, context, allocator, local_context, local_allocator);

        for (auto& tensor : results) {
            switch (tensor.dtype()) {
            case Tensor::DType<float>::value:
            {
                auto data = context.read<float>(tensor);
                print_tensor_like(data, tensor.shape());
                break;
            }

            case Tensor::DType<int8_t>::value:
            {
                auto data = context.read<int8_t>(tensor);
                print_tensor_like(data, tensor.shape());
                break;
            }

            case Tensor::DType<int16_t>::value:
            {
                auto data = context.read<int16_t>(tensor);
                print_tensor_like(data, tensor.shape());
                break;
            }

            case Tensor::DType<int32_t>::value:
            {
                auto data = context.read<int32_t>(tensor);
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
