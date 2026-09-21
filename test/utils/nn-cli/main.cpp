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

    virtual Computation<std::vector<Tensor>> compute(Context& context) {
        if (args_.get(0) == "Linear") {
            Computation<Tensor> computation(context);

            auto in_features = args_.get_one<int64_t>("--in_features");
            auto out_features = args_.get_one<int64_t>("--out_features");
            auto bias = args_.get_optional<bool>("--bias").value_or(true);
            auto x = args_.get_one<Tensor>("--x", {computation.desc()->state_ctx()});

            Linear model(in_features, out_features, bias);

            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return model.forward(scope, x);
            });

            return Computation<Tensor>::all(result);
        }
        
        if (args_.get(0) == "SiLU") {
            Computation<Tensor> computation(context);

            auto x = args_.get_one<Tensor>("--x", {computation.desc()->state_ctx()});

            SiLU model;

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return model.forward(scope.context(), x);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "RMSNorm") {
            Computation<Tensor> computation(context);

            auto dim = args_.get_one<int64_t>("--dim");
            auto eps = args_.get_optional<float>("--eps").value_or(1e-5f);
            auto elementwise_affine = args_.get_optional<bool>("--elementwise_affine").value_or(true);
            auto x = args_.get_one<Tensor>("--x", {computation.desc()->state_ctx()});

            RMSNorm model(dim, eps, elementwise_affine);

            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return model.forward(scope.context(), x);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "LayerNorm") {
            Computation<Tensor> computation(context);

            auto dim = args_.get_one<int64_t>("--dim");
            auto eps = args_.get_optional<float>("--eps").value_or(1e-5f);
            auto elementwise_affine = args_.get_optional<bool>("--elementwise_affine").value_or(true);
            auto bias = args_.get_optional<bool>("--bias").value_or(true);
            auto x = args_.get_one<Tensor>("--x", {computation.desc()->state_ctx()});

            LayerNorm model(dim, eps, elementwise_affine, bias);

            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return model.forward(scope.context(), x);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "GroupNorm") {
            Computation<Tensor> computation(context);

            auto num_groups = args_.get_one<int64_t>("--num_groups");
            auto num_channels = args_.get_one<int64_t>("--num_channels");
            auto eps = args_.get_optional<float>("--eps").value_or(1e-5f);
            auto affine = args_.get_optional<bool>("--affine").value_or(true);
            auto bias = args_.get_optional<bool>("--bias").value_or(true);
            auto input = args_.get_one<Tensor>("--input", {computation.desc()->state_ctx()});

            GroupNorm model(num_groups, num_channels, eps, affine, bias);

            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return model.forward(scope.context(), input);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "Conv2d") {
            Computation<Tensor> computation(context);

            auto in_channels = args_.get_one<int64_t>("--in_channels");
            auto out_channels = args_.get_one<int64_t>("--out_channels");
            auto kernel_size = args_.get_one<int64_t>("--kernel_size");
            auto stride = args_.get_optional<int64_t>("--stride").value_or(1);
            auto padding = args_.get_optional<int64_t>("--padding").value_or(0);
            auto bias = args_.get_optional<bool>("--bias").value_or(true);
            auto x = args_.get_one<Tensor>("--x", {computation.desc()->state_ctx()});

            Conv2d model(in_channels, out_channels, kernel_size, stride, padding, bias);

            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return model.forward(scope.context(), x);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "FlashAttention") {
            Computation<Tensor> computation(context);

            auto q = args_.get_one<Tensor>("--q", {computation.desc()->state_ctx()});
            auto k = args_.get_one<Tensor>("--k", {computation.desc()->state_ctx()});
            auto v = args_.get_one<Tensor>("--v", {computation.desc()->state_ctx()});
            auto mask = args_.get_optional<Tensor>("--mask", {computation.desc()->state_ctx()});

            FlashAttentionOp attention;

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return attention(scope.context(), q, k, v, mask);
            });

            return Computation<Tensor>::all(result);
        }

        if (args_.get(0) == "Embedding") {
            Computation<Tensor> computation(context);

            auto num_embeddings = args_.get_one<int64_t>("--num_embeddings");
            auto embedding_dim = args_.get_one<int64_t>("--embedding_dim");
            auto padding_idx = args_.get_optional<int64_t>("--padding_idx");
            auto input = args_.get_one<Tensor>("--input", {computation.desc()->state_ctx(), Tensor::DType<int32_t>::value});

            Embedding model(num_embeddings, embedding_dim, padding_idx);

            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return model.forward(scope.context(), input);
            });

            return Computation<Tensor>::all(result);
        }
        
        if (args_.get(0) == "ShardedMLP") {
            Computation<Tensor> computation(context);
            MLP<SiLU> model;

            {
                CreateRandomParametersVisitor create_parameters(context);
                RethrowVisitor visitor(create_parameters);
                model.accept(visitor);
                visitor.rethrow();
            }

            auto x = Tensor::empty<float>(Tensor::Shape{2, 3, 4, 8}).input();

            auto result = computation.scope([&](Scope scope) -> Tensor {
                return model.forward(scope.context(), x);
            });

            return Computation<Tensor>::all(result);
        }

        throw std::runtime_error("Uknown command: " + args_.get(0));
    }
};

int main(int argc, char** argv) {
    TestNnCLI cli(argc, argv);
    return cli.main();
}
