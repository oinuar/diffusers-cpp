#include <vector>
#include <unordered_set>
#include <array>
#include <iostream>
#include "ggml/Compute.hpp"
#include "ggml/Graph.hpp"
#include "ggml/Computation.hpp"
#include "ggml/Context.hpp"
#include "ggml/Backend.hpp"
#include "ggml/Scheduler.hpp"
#include "nn/Parameter.hpp"
#include "nn/Visitor.hpp"
#include "../ArgumentParser.hpp"

#include "nn/Linear.hpp"
#include "nn/SiLU.hpp"
#include "nn/RMSNorm.hpp"
#include "nn/LayerNorm.hpp"

struct Node {
    std::string op;
    std::string type;
    Tensor::Shape shape;
};

void traverse(
    ggml_tensor* tensor,
    std::vector<Node>& out,
    std::unordered_set<ggml_tensor*>& visited)
{
    if (!tensor)
        return;

    if (!visited.insert(tensor).second)
        return;

    for (int i = 0; i < GGML_MAX_SRC; ++i)
        traverse(tensor->src[i], out, visited);

    Tensor::Shape shape(ggml_n_dims(tensor));

    for (auto i = 0; i < shape.rank(); ++i)
        shape[i] = tensor->ne[i];

    std::string op(ggml_op_name(tensor->op));

    if (tensor->op == GGML_OP_UNARY)
        op = ggml_unary_op_name(ggml_get_unary_op(tensor));

    std::string type(ggml_type_name(tensor->type));

    out.push_back({ op, type, shape });
}

void print_dag(ggml_tensor* root)
{
    std::vector<Node> graph;
    std::unordered_set<ggml_tensor*> visited;

    traverse(root, graph, visited);

    for (auto& node : graph)
        std::cout << node.op << " " << node.type << " " << node.shape.to_string() << std::endl;
}

class CreateParametersVisitor : public Visitor {
public:
    CreateParametersVisitor(Context& ctx) : ctx_(ctx) {}

    virtual void visit(Parameter& parameter, std::vector<std::string>) {
        parameter.set(Tensor::empty(*ctx_, parameter.shape()));
    }

private:
    Context& ctx_;
};

Tensor build(Context& ctx, const ArgumentParser& args) {
    if (args.get(0) == "linear") {
        auto in_features = args.get_one<int64_t>("--in_features");
        auto out_features = args.get_one<int64_t>("--out_features");
        auto bias = args.get_optional<bool>("--bias").value_or(true);
        auto input = args.get_one<Tensor::Shape>("--input");

        Linear model(in_features, out_features, bias);

        CreateParametersVisitor visitor(ctx);
        model.accept(visitor);

        auto x = Tensor::empty(*ctx, input);
        auto y = model.forward(*ctx, x);

        return y;
    }
    
    if (args.get(0) == "silu") {
        auto input = args.get_one<Tensor::Shape>("--input");

        SiLU model;

        auto x = Tensor::empty(*ctx, input);
        auto y = model.forward(*ctx, x);

        return y;
    }

    if (args.get(0) == "rmsnorm") {
        auto dim = args.get_one<int64_t>("--dim");
        auto input = args.get_one<Tensor::Shape>("--input");
        auto eps = args.get_optional<float>("--eps").value_or(1e-5f);
        auto elementwise_affine = args.get_optional<bool>("--elementwise_affine").value_or(true);

        RMSNorm model(dim, eps, elementwise_affine);

        CreateParametersVisitor visitor(ctx);
        model.accept(visitor);

        auto x = Tensor::empty(*ctx, input);
        auto y = model.forward(*ctx, x);

        return y;
    }

    if (args.get(0) == "layernorm") {
        auto dim = args.get_one<int64_t>("--dim");
        auto input = args.get_one<Tensor::Shape>("--input");
        auto eps = args.get_optional<float>("--eps").value_or(1e-5f);
        auto elementwise_affine = args.get_optional<bool>("--elementwise_affine").value_or(true);
        auto bias = args.get_optional<bool>("--bias").value_or(true);

        LayerNorm model(dim, eps, elementwise_affine, bias);

        CreateParametersVisitor visitor(ctx);
        model.accept(visitor);

        auto x = Tensor::empty(*ctx, input);
        auto y = model.forward(*ctx, x);

        return y;
    }

    throw std::runtime_error("Uknown command: " + args.get(0));
}

int main(int argc, char** argv) {
    ArgumentParser args(argc, argv);
    Arena arena;
    Context ctx(arena);

    auto tensor = build(ctx, args);

    print_dag(*tensor);

    return EXIT_SUCCESS;
}
