#include "../TestCLI.hpp"
#include "nn/RethrowVisitor.hpp"

#include "nn/Linear.hpp"
#include "nn/SiLU.hpp"
#include "nn/Embedding.hpp"
#include "nn/modules/normalization/RMSNorm.hpp"
#include "nn/modules/normalization/LayerNorm.hpp"
#include "nn/modules/normalization/GroupNorm.hpp"
#include "nn/modules/conv/Conv2d.hpp"
#include "nn/attention/ScaledDotProductAttention.hpp"
#include "nn/attention/FlashAttentionOp.hpp"

#include <numeric>

class TestNnCLI : public TestCLI {
public:
    TestNnCLI(int argc, char** argv) : TestCLI(argc, argv) {}

    template <class Act>
    class MLP : public Module {
    public:
        MLP() {
            modules["fc1"] = std::make_shared<Linear>(8, 16);
            modules["act"] = std::make_shared<Act>();
            modules["fc2"] = std::make_shared<Linear>(16, 8);
        }

        Tensor forward(Scope scope, Tensor x) {
            x = std::static_pointer_cast<Linear>(modules["fc1"])->forward(scope, x);
            x = std::static_pointer_cast<Act>(modules["act"])->forward(scope, x);
            return std::static_pointer_cast<Linear>(modules["fc2"])->forward(scope, x);
        }
    };

    // random models; the planner does not care about values.)
    class CreateRandomParametersVisitor : public Visitor {
    public:
        CreateRandomParametersVisitor(Scope scope) : scope_(scope) {}

        void visit(Parameter& parameter, std::vector<std::string> path) override {
            std::string name;
            for (size_t i = 0; i < path.size(); ++i)
                name += (i ? "." : "") + path[i];

            Tensor::Shape shape = parameter.shape();
            Tensor t = Tensor(scope_.runtime().new_tensor(GGML_TYPE_F32, (int)shape.rank(), shape.data()), shape);
            ggml_set_name(*t, name.c_str());
            parameter.set(std::move(t));
        }

    private:
        Scope scope_;
    };

    virtual std::vector<Tensor> compute(Allocator& allocator, Scheduler& scheduler, Context& context, Context& local_context) {
        if (args_.get(0) == "Linear") {
            Scope scope(local_context, allocator.runtime());

            auto in_features = args_.get_one<int64_t>("--in_features");
            auto out_features = args_.get_one<int64_t>("--out_features");
            auto bias = args_.get_optional<bool>("--bias").value_or(true);
            auto x = args_.get_one<Tensor>("--x", {scope.context()});

            Linear model(in_features, out_features, bias);

            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();
            
            auto output = model.forward(scope.context(), x);

            Graph graph(scheduler, scope.context(), {output});
            Computation computation(allocator, graph, {&context, &scope.context()});
            return computation().results();
        }
        
        if (args_.get(0) == "SiLU") {
            Scope scope(local_context, allocator.runtime());

            auto x = args_.get_one<Tensor>("--x", {scope.context()});

            SiLU model;

            auto output = model.forward(scope.context(), x);

            Graph graph(scheduler, scope.context(), {output});
            Computation computation(allocator, graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "RMSNorm") {
            Scope scope(local_context, allocator.runtime());

            auto dim = args_.get_one<int64_t>("--dim");
            auto eps = args_.get_optional<float>("--eps").value_or(1e-5f);
            auto elementwise_affine = args_.get_optional<bool>("--elementwise_affine").value_or(true);
            auto x = args_.get_one<Tensor>("--x", {scope.context()});

            RMSNorm model(dim, eps, elementwise_affine);

            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto output = model.forward(scope.context(), x);

            Graph graph(scheduler, scope.context(), {output});
            Computation computation(allocator, graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "LayerNorm") {
            Scope scope(local_context, allocator.runtime());

            auto dim = args_.get_one<int64_t>("--dim");
            auto eps = args_.get_optional<float>("--eps").value_or(1e-5f);
            auto elementwise_affine = args_.get_optional<bool>("--elementwise_affine").value_or(true);
            auto bias = args_.get_optional<bool>("--bias").value_or(true);
            auto x = args_.get_one<Tensor>("--x", {scope.context()});

            LayerNorm model(dim, eps, elementwise_affine, bias);

            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto output = model.forward(scope.context(), x);

            Graph graph(scheduler, scope.context(), {output});
            Computation computation(allocator, graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "GroupNorm") {
            Scope scope(local_context, allocator.runtime());

            auto num_groups = args_.get_one<int64_t>("--num_groups");
            auto num_channels = args_.get_one<int64_t>("--num_channels");
            auto eps = args_.get_optional<float>("--eps").value_or(1e-5f);
            auto affine = args_.get_optional<bool>("--affine").value_or(true);
            auto bias = args_.get_optional<bool>("--bias").value_or(true);
            auto input = args_.get_one<Tensor>("--input", {scope.context()});

            GroupNorm model(num_groups, num_channels, eps, affine, bias);

            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto output = model.forward(scope.context(), input);

            Graph graph(scheduler, scope.context(), {output});
            Computation computation(allocator, graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "Conv2d") {
            Scope scope(local_context, allocator.runtime());

            auto in_channels = args_.get_one<int64_t>("--in_channels");
            auto out_channels = args_.get_one<int64_t>("--out_channels");
            auto kernel_size = args_.get_one<int64_t>("--kernel_size");
            auto stride = args_.get_optional<int64_t>("--stride").value_or(1);
            auto padding = args_.get_optional<int64_t>("--padding").value_or(0);
            auto bias = args_.get_optional<bool>("--bias").value_or(true);
            auto x = args_.get_one<Tensor>("--x", {scope.context()});

            Conv2d model(in_channels, out_channels, kernel_size, stride, padding, bias);

            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto output = model.forward(scope.context(), x);

            Graph graph(scheduler, scope.context(), {output});
            Computation computation(allocator, graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "FlashAttention") {
            Scope scope(local_context, allocator.runtime());

            auto q = args_.get_one<Tensor>("--q", {scope.context()});
            auto k = args_.get_one<Tensor>("--k", {scope.context()});
            auto v = args_.get_one<Tensor>("--v", {scope.context()});
            auto mask = args_.get_optional<Tensor>("--mask", {scope.context()});

            FlashAttentionOp attention;

            auto output = attention(context, q, k, v, mask);

            Graph graph(scheduler, scope.context(), {output});
            Computation computation(allocator, graph, {&context, &scope.context()});
            return computation().results();
        }

        if (args_.get(0) == "Embedding") {
            Scope scope(local_context, allocator.runtime());

            auto num_embeddings = args_.get_one<int64_t>("--num_embeddings");
            auto embedding_dim = args_.get_one<int64_t>("--embedding_dim");
            auto padding_idx = args_.get_optional<int64_t>("--padding_idx");
            auto input = args_.get_one<Tensor>("--input", {scope.context()});

            Embedding model(num_embeddings, embedding_dim, padding_idx);

            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto output = model.forward(scope.context(), input);

            Graph graph(scheduler, scope.context(), {output});
            Computation computation(allocator, graph, {&context, &scope.context()});
            return computation().results();
        }
        
        if (args_.get(0) == "ShardedMLP") {
            Scope scope(local_context, allocator.runtime());
            MLP<SiLU> model;

            {
                CreateRandomParametersVisitor create_parameters(context);
                RethrowVisitor visitor(create_parameters);
                model.accept(visitor);
                visitor.rethrow();
            }

            auto x = Tensor::empty<float>(Tensor::Shape{2, 3, 4, 8}).input();

            auto output = model.forward(scope.context(), x);

            Graph graph(scheduler, scope.context(), {output});
            Computation computation(allocator, graph, {&context, &scope.context()});
            return computation().results();
        }

        throw std::runtime_error("Uknown command: " + args_.get(0));
    }
};

int main(int argc, char** argv) {
    TestNnCLI cli(argc, argv);
    return cli.main();
}
