#include <functional>
#include "GGMLCompute.hpp"
#include "GGMLGraph.hpp"
#include "GGMLComputation.hpp"
#include "GGMLContext.hpp"
#include "GGMLBackend.hpp"
#include "GGMLScheduler.hpp"
#include "../ArgumentParser.hpp"

static void print_tensor(const Tensor& t) {
    auto shape   = t.shape();
    double* data = nullptr;

    int64_t idx = 0;

    std::function<void(int)> recurse = [&](int dim) {
        if (dim == shape.rank() - 1) {
            for (int64_t i = 0; i < shape[dim]; ++i) {
                if (idx++ > 0)
                    std::cout << ",";
                double v = data[idx - 1];

                if (v == 0.0 && v < 0)
                    std::cout << "-0.0";
                else if (std::isinf(v))
                    std::cout << (v > 0 ? "Infinity" : "-Infinity");
                else if (std::isnan(v))
                    std::cout << "NaN";
                else
                    printf("%.6g", v);
            }
        } else {
            std::cout << "[";
            for (int64_t i = 0; i < shape[dim]; ++i) {
                recurse(dim + 1);
            }
            std::cout << "]";
        }
    };

    recurse(0);
    std::cout << std::endl;
}

// --- unary ops ---
/*int neg_fn(ArgumentParser& a) {
    auto t = a.get<Tensor>("--this");
    print_tensor(-t);
    return EXIT_SUCCESS;
}
int abs_fn(ArgumentParser& a) {
    auto t = a.get<Tensor>("--this");
    print_tensor(::abs(t));
    return EXIT_SUCCESS;
}
int sqrt_fn(ArgumentParser& a) {
    auto t = a.get<Tensor>("--this");
    print_tensor(::sqrt(t));
    return EXIT_SUCCESS;
}
int rsqrt_fn(ArgumentParser& a) {
    auto t = a.get<Tensor>("--this");
    print_tensor(::rsqrt(t));
    return EXIT_SUCCESS;
}
int exp_fn(ArgumentParser& a) {
    auto t = a.get<Tensor>("--this");
    print_tensor(::exp(t));
    return EXIT_SUCCESS;
}
int log_fn(ArgumentParser& a) {
    auto t = a.get<Tensor>("--this");
    print_tensor(::log(t));
    return EXIT_SUCCESS;
}
int sin_fn(ArgumentParser& a) {
    auto t = a.get<Tensor>("--this");
    print_tensor(::sin(t));
    return EXIT_SUCCESS;
}
int cos_fn(ArgumentParser& a) {
    auto t = a.get<Tensor>("--this");
    print_tensor(::cos(t));
    return EXIT_SUCCESS;
}

// --- binary tensor ops ---
int add(ArgumentParser& a) {
    auto l = a.get<Tensor>("--lhs"), r = a.get<Tensor>("--rhs");
    print_tensor(l + r);
    return EXIT_SUCCESS;
}
int sub(ArgumentParser& a) {
    auto l = a.get<Tensor>("--lhs"), r = a.get<Tensor>("--rhs");
    print_tensor(l - r);
    return EXIT_SUCCESS;
}
int mul_fn(ArgumentParser& a) {
    auto l = a.get<Tensor>("--lhs"), r = a.get<Tensor>("--rhs");
    print_tensor(l * r);
    return EXIT_SUCCESS;
}
int div_fn(ArgumentParser& a) {
    auto l = a.get<Tensor>("--lhs"), r = a.get<Tensor>("--rhs");
    print_tensor(l / r);
    return EXIT_SUCCESS;
}

// --- tensor op scalar ---
int add_scalar(ArgumentParser& a) {
    auto t  = a.get<Tensor>("--this");
    float v = a.get<float>("--scalar");
    print_tensor(t + v);
    return EXIT_SUCCESS;
}
int sub_scalar(ArgumentParser& a) {
    auto t  = a.get<Tensor>("--this");
    float v = a.get<float>("--scalar");
    print_tensor(t - v);
    return EXIT_SUCCESS;
}
int mul_scalar(ArgumentParser& a) {
    auto t  = a.get<Tensor>("--this");
    float v = a.get<float>("--scalar");
    print_tensor(t * v);
    return EXIT_SUCCESS;
}
int div_scalar(ArgumentParser& a) {
    auto t  = a.get<Tensor>("--this");
    float v = a.get<float>("--scalar");
    print_tensor(t / v);
    return EXIT_SUCCESS;
}

// --- scalar op tensor ---
int scalar_add(ArgumentParser& a) {
    float v = a.get<float>("--scalar");
    auto t  = a.get<Tensor>("--this");
    print_tensor(v + t);
    return EXIT_SUCCESS;
}
int scalar_sub(ArgumentParser& a) {
    float v = a.get<float>("--scalar");
    auto t  = a.get<Tensor>("--this");
    print_tensor(v - t);
    return EXIT_SUCCESS;
}
int scalar_mul(ArgumentParser& a) {
    float v = a.get<float>("--scalar");
    auto t  = a.get<Tensor>("--this");
    print_tensor(v * t);
    return EXIT_SUCCESS;
}
int scalar_div(ArgumentParser& a) {
    float v = a.get<float>("--scalar");
    auto t  = a.get<Tensor>("--this");
    print_tensor(v / t);
    return EXIT_SUCCESS;
}

// --- shape / dim ops ---
int reshape(ArgumentParser& a) {
    auto t = a.get<Tensor>("--this");
    print_tensor(t.reshape(a.get<Tensor::Shape>("--shape")));
    return EXIT_SUCCESS;
}
int permute(ArgumentParser& a) {
    auto t = a.get<Tensor>("--this");
    print_tensor(t.permute(a.get<Tensor::Shape>("--order")));
    return EXIT_SUCCESS;
}
int squeeze(ArgumentParser& a) {
    auto t = a.get<Tensor>("--this");
    print_tensor(t.squeeze(a.get<int>("--dim")));
    return EXIT_SUCCESS;
}
int unsqueeze(ArgumentParser& a) {
    auto t = a.get<Tensor>("--this");
    print_tensor(t.unsqueeze(a.get<int>("--dim")));
    return EXIT_SUCCESS;
}
int flatten(ArgumentParser& a) {
    auto t = a.get<Tensor>("--this");
    print_tensor(t.flatten(a.get<int>("--start_dim"), a.get<int>("--end_dim")));
    return EXIT_SUCCESS;
}
int unflatten(ArgumentParser& a) {
    auto t = a.get<Tensor>("--this");
    print_tensor(t.unflatten(a.get<int64_t>("--dim"), a.get<Tensor::Shape>("--shape")));
    return EXIT_SUCCESS;
}
int narrow(ArgumentParser& a) {
    auto t = a.get<Tensor>("--this");
    print_tensor(t.narrow(a.get<int>("--dim"), a.get<int64_t>("--start"), a.get<int64_t>("--length")));
    return EXIT_SUCCESS;
}

// --- slice op[] ---
int slice(ArgumentParser& a) {
    auto t          = a.get<Tensor>("--this");
    std::string raw = a.get<std::string>("--slice");
    std::vector<Tensor::Slice> slices;

    size_t pos = 0;
    while (pos <= raw.size()) {
        auto end = raw.find('|', pos);
        if (end == std::string::npos)
            end = raw.size();
        auto part = raw.substr(pos, end - pos);

        if (part == ":")
            slices.push_back(Tensor::Slice::all());
        else if (part == "None")
            slices.push_back(Tensor::Slice::none());
        else {
            auto c = part.find(':');
            if (c != std::string::npos) {
                auto start    = (c == 0) ? std::optional<int64_t>(std::stoi(part.substr(0, c))) : std::nullopt;
                auto stop_str = part.substr(c + 1);
                auto s2       = stop_str.find(':');
                std::optional<int64_t> stop;
                int step = 1;
                if (s2 != std::string::npos) {
                    stop = std::stoi(stop_str.substr(0, s2));
                    step = std::stoi(stop_str.substr(s2 + 1));
                } else if (!stop_str.empty())
                    stop = std::stoi(stop_str);
                slices.push_back(Tensor::Slice::range(start, stop, step));
            } else {
                slices.push_back(Tensor::Slice::index(std::stoi(part)));
            }
        }

        if (end >= raw.size())
            break;
        pos = end + 1;
    }

    print_tensor(t[slices]);
    return EXIT_SUCCESS;
}

// --- reductions ---
int sum(ArgumentParser& a) {
    auto t = a.get<Tensor>("--this");
    print_tensor(t.sum());
    return EXIT_SUCCESS;
}
int mean(ArgumentParser& a) {
    auto t        = a.get<Tensor>("--this");
    auto dim = a.get<int>("--dim");
    auto keepdims = (bool)a.get<int>("--keepdims");
    print_tensor(t.mean(dim, keepdims));
    return EXIT_SUCCESS;
}

// --- cast ---
int cast(ArgumentParser& a) {
    auto t = a.get<Tensor>("--this");
    print_tensor(t.to((ggml_type)a.get<int>("--type")));
    return EXIT_SUCCESS;
}

// --- expand / chunk / split ---
int expand(ArgumentParser& a) {
    auto t = a.get<Tensor>("--this");
    print_tensor(t.expand(a.get<Tensor::Shape>("--shape")));
    return EXIT_SUCCESS;
}

int chunk(ArgumentParser& a) {
    auto t  = a.get<Tensor>("--this");
    auto cs = t.chunk(a.get<int>("--n"), a.get<int>("--dim"));
    std::cout << "[";
    for (size_t i = 0; i < cs.size(); ++i) {
        if (i > 0)
            std::cout << ",";
        print_tensor(cs[i]);
    }
    std::cout << "]" << std::endl;
    return EXIT_SUCCESS;
}

int split(ArgumentParser& a) {
    auto t  = a.get<Tensor>("--this");
    auto cs = t.split(a.get<int64_t>("--size"), a.get<int>("--dim"));
    std::cout << "[";
    for (size_t i = 0; i < cs.size(); ++i) {
        if (i > 0)
            std::cout << ",";
        print_tensor(cs[i]);
    }
    std::cout << "]" << std::endl;
    return EXIT_SUCCESS;
}

int split_with_sizes(ArgumentParser& a) {
    auto t         = a.get<Tensor>("--this");
    auto sizes_raw = a.get<std::string>("--sizes");
    std::vector<int64_t> sizes;
    {
        size_t p = 0;
        while (p <= sizes_raw.size()) {
            auto e = sizes_raw.find(',', p);
            if (e == std::string::npos)
                e = sizes_raw.size();
            if (p < e)
                sizes.push_back(std::stoll(sizes_raw.substr(p, e - p)));
            if (e >= sizes_raw.size())
                break;
            p = e + 1;
        }
    }
    auto cs = t.split_with_sizes(sizes, a.get<int>("--dim"));
    std::cout << "[";
    for (size_t i = 0; i < cs.size(); ++i) {
        if (i > 0)
            std::cout << ",";
        print_tensor(cs[i]);
    }
    std::cout << "]" << std::endl;
    return EXIT_SUCCESS;
}

// --- math ---
int pow_fn(ArgumentParser& a) {
    auto t = a.get<Tensor>("--this");
    print_tensor(t.pow(a.get<float>("--exponent")));
    return EXIT_SUCCESS;
}
int clip_fn(ArgumentParser& a) {
    auto t = a.get<Tensor>("--this");
    print_tensor(t.clip(a.get<float>("--min"), a.get<float>("--max")));
    return EXIT_SUCCESS;
}

// --- cat (static) ---
int cat_fn(ArgumentParser& a) {
    auto tensors = a.get<std::vector<Tensor>>("--tensor");
    print_tensor(Tensor::cat(tensors, a.get<int>("--dim")));
    return EXIT_SUCCESS;
}

// --- introspection ---
Tensor contiguous_fn(ArgumentParser& a) {
    auto t = a.get<Tensor>("--this");
    return t.contiguous();
}*/

class TensorCLI : public GGMLCompute {
public:
    TensorCLI(int argc, char** argv) : args_(argc, argv) {}

    Tensor build(GGMLContext& ctx) {

        // factory
        if (args_.command() == "zeros") {
            auto shape = args_.get<Tensor::Shape>("--shape");

            return Tensor::zeros(*ctx, shape);
        }

        if (args_.command() == "ones") {
            auto shape = args_.get<Tensor::Shape>("--shape");

            return Tensor::ones(*ctx, shape);
        }

        // unary ops
        if (args_.command() == "neg") {
            auto self = get_tensor(ctx, "--this");
            
            return -self;
        }

        /*if (args_.command() == "abs") return abs_fn(args_);
        if (args_.command() == "sqrt") return sqrt_fn(args_);
        if (args_.command() == "rsqrt") return rsqrt_fn(args_);
        if (args_.command() == "exp") return exp_fn(args_);
        if (args_.command() == "log") return log_fn(args_);
        if (args_.command() == "sin") return sin_fn(args_);
        if (args_.command() == "cos") return cos_fn(args_);*/

        // binary tensor ops
        if (args_.command() == "add") {
            auto lhs = get_tensor(ctx, "--lhs");
            auto rhs = get_tensor(ctx, "--rhs");

            return lhs + rhs;
        }

        /*if (args_.command() == "sub") return sub(args_);
        if (args_.command() == "mul") return mul_fn(args_);
        if (args_.command() == "div") return div_fn(args_);

        // tensor op scalar
        if (args_.command() == "add_scalar") return add_scalar(args_);
        if (args_.command() == "sub_scalar") return sub_scalar(args_);
        if (args_.command() == "mul_scalar") return mul_scalar(args_);
        if (args_.command() == "div_scalar") return div_scalar(args_);

        // scalar op tensor
        if (args_.command() == "scalar_add") return scalar_add(args_);
        if (args_.command() == "scalar_sub") return scalar_sub(args_);
        if (args_.command() == "scalar_mul") return scalar_mul(args_);
        if (args_.command() == "scalar_div") return scalar_div(args_);

        // shape / dim ops
        if (args_.command() == "reshape") return reshape(args_);
        if (args_.command() == "permute") return permute(args_);
        if (args_.command() == "squeeze") return squeeze(args_);
        if (args_.command() == "unsqueeze") return unsqueeze(args_);
        if (args_.command() == "flatten") return flatten(args_);
        if (args_.command() == "unflatten") return unflatten(args_);
        if (args_.command() == "narrow") return narrow(args_);

        // slice op[]
        if (args_.command() == "slice") return slice(args_);

        // reductions
        if (args_.command() == "sum") return sum(args_);
        if (args_.command() == "mean") return mean(args_);

        // cast / expand / chunk / split
        if (args_.command() == "cast") return cast(args_);
        if (args_.command() == "expand") return expand(args_);
        if (args_.command() == "chunk") return chunk(args_);
        if (args_.command() == "split_with_sizes") return split_with_sizes(args_);
        if (args_.command() == "split") return split(args_);

        // math
        if (args_.command() == "pow") return pow_fn(args_);
        if (args_.command() == "clip") return clip_fn(args_);

        // cat (static)
        if (args_.command() == "cat") return cat_fn(args_);

        // introspection
        if (args_.command() == "contiguous") return contiguous_fn(args_);*/

        throw std::runtime_error("Uknown command: " + args_.command());
    }

    void compute(GGMLGraph& graph) {
        GGMLComputation computation(graph);

        for (auto& [tensor, data] : inputs_)
            computation.load(tensor, reinterpret_cast<std::byte*>(data.data()));

        computation.execute();

        auto [shape, data] = computation.read<float>(-1);

        // TODO: shape data printing to stdout

        for (auto& value : data)
            std::cout << std::to_string(value) << std::endl;
    }

private:
    ArgumentParser args_;
    std::vector<std::pair<Tensor, std::vector<float>>> inputs_;

    Tensor get_tensor(GGMLContext& ctx, std::string_view option) {
        auto [shape, data] = args_.get<std::pair<Tensor::Shape, std::vector<float>>>(option);
        auto tensor = Tensor::empty(*ctx, GGML_TYPE_F32, shape);

        inputs_.push_back({tensor, data});

        return tensor;
    }
};

int main(int argc, char** argv) {
    ggml_time_init();
    ggml_log_set([](ggml_log_level, const char* text, void*) { std::cerr << text; }, nullptr);

    ggml_backend_load_all();

    GGMLBackend cpu(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGMLScheduler scheduler({*cpu});

    TensorCLI tensor_cli(argc, argv);

    auto graph = scheduler.plan(tensor_cli);
    tensor_cli.compute(graph);

    return EXIT_SUCCESS;
}
