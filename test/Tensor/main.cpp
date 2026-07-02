#include "../ArgumentParser.hpp"

void print_tensor_recursive(
    const float* data,
    const std::vector<int64_t>& shape,
    int dim,
    int64_t offset,
    std::ostream& os)
{
    if (dim == (int)shape.size() - 1) {
        os << "[";

        for (int64_t i = 0; i < shape[dim]; ++i) {
            os << data[offset + i];
            if (i + 1 < shape[dim]) os << ", ";
        }

        os << "]";
        return;
    }

    os << "[";

    int64_t stride = 1;
    for (int i = dim + 1; i < (int)shape.size(); ++i)
        stride *= shape[i];

    for (int64_t i = 0; i < shape[dim]; ++i) {
        print_tensor_recursive(
            data,
            shape,
            dim + 1,
            offset + i * stride,
            os);

        if (i + 1 < shape[dim]) os << ", ";
    }

    os << "]";
}

void print_output(const Tensor& t)
{
    if (!*t) {
        std::cout << "null\n";
        return;
    }

    ggml_tensor* gt = *t;

    int ndim = ggml_n_dims(gt);

    std::vector<int64_t> shape(ndim);
    for (int i = 0; i < ndim; ++i)
        shape[i] = gt->ne[i];

    const float* data = (const float*) ggml_get_data(gt);

    print_tensor_recursive(data, shape, 0, 0, std::cout);

    std::cout << "\n";
}

int command_reshape(ArgumentParser& args)
{
    auto input = args.get<Tensor>("--this");
    auto shape = args.get<Tensor::Shape>("--shape");

    auto output = input.reshape(shape);

    // TODO: print output to stdout

    return EXIT_SUCCESS;
}

int command_narrow(ArgumentParser& args)
{
    auto input = args.get<Tensor>("--this");
    auto dim = args.get<int>("--dim");
    auto start = args.get<int64_t>("--start");
    auto len = args.get<int64_t>("--length");

    auto output = input.narrow(dim, start, len);

    // TODO: print output to stdout

    return EXIT_SUCCESS;
}

int command_add(ArgumentParser& args)
{
    auto lhs = args.get<Tensor>("--lhs");
    auto rhs = args.get<Tensor>("--rhs");

    //auto output = lhs + rhs;

    // TODO: print output to stdout

    return EXIT_SUCCESS;
}

int main(int argc, char** argv)
{
    try {
        ArgumentParser args(argc, argv);

        const auto command = args.command();

        if (command == "reshape")
            return command_reshape(args);

        if (command == "narrow")
            return command_narrow(args);

        if (command == "add")
            return command_add(args);

        throw std::runtime_error("Unknown command: " + command);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
