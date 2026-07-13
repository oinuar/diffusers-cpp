#include "../TestCLI.hpp"
#include "nn/Parameter.hpp"
#include "nn/Visitor.hpp"
#include "nn/Linear.hpp"
#include "nn/SiLU.hpp"
#include "nn/RMSNorm.hpp"
#include "nn/LayerNorm.hpp"
#include "nn/AdaLayerNormContinuous.hpp"
#include <numeric>

class TestNnCLI : public TestCLI {
public:
    TestNnCLI(int argc, char** argv) : TestCLI(argc, argv) {}

    virtual Plan build(Context& ctx) {

        if (args_.get(0) == "Linear") {
            auto in_features = args_.get_one<int64_t>("--in_features");
            auto out_features = args_.get_one<int64_t>("--out_features");
            auto bias = args_.get_optional<bool>("--bias").value_or(true);
            auto x = args_.get_one<Tensor>("--x", {ctx, inputs_});

            Linear model(in_features, out_features, bias);

            CreateParametersVisitor visitor(ctx, inputs_, args_);
            model.accept(visitor);

            return model.forward(*ctx, x);
        }
        
        if (args_.get(0) == "SiLU") {
            auto x = args_.get_one<Tensor>("--x", {ctx, inputs_});

            SiLU model;

            return model.forward(*ctx, x);
        }

        if (args_.get(0) == "RMSNorm") {
            auto dim = args_.get_one<int64_t>("--dim");
            auto eps = args_.get_optional<float>("--eps").value_or(1e-5f);
            auto elementwise_affine = args_.get_optional<bool>("--elementwise_affine").value_or(true);
            auto x = args_.get_one<Tensor>("--x", {ctx, inputs_});

            RMSNorm model(dim, eps, elementwise_affine);

            CreateParametersVisitor visitor(ctx, inputs_, args_);
            model.accept(visitor);

            return model.forward(*ctx, x);
        }

        if (args_.get(0) == "LayerNorm") {
            auto dim = args_.get_one<int64_t>("--dim");
            auto eps = args_.get_optional<float>("--eps").value_or(1e-5f);
            auto elementwise_affine = args_.get_optional<bool>("--elementwise_affine").value_or(true);
            auto bias = args_.get_optional<bool>("--bias").value_or(true);
            auto x = args_.get_one<Tensor>("--x", {ctx, inputs_});

            LayerNorm model(dim, eps, elementwise_affine, bias);

            CreateParametersVisitor visitor(ctx, inputs_, args_);
            model.accept(visitor);

            return model.forward(*ctx, x);
        }

        if (args_.get(0) == "AdaLayerNormContinuous") {
            auto embedding_dim = args_.get_one<int64_t>("--embedding_dim");
            auto conditioning_embedding_dim = args_.get_one<int64_t>("--conditioning_embedding_dim");
            auto elementwise_affine = args_.get_optional<bool>("--elementwise_affine").value_or(true);
            auto eps = args_.get_optional<float>("--eps").value_or(1e-5f);
            auto bias = args_.get_optional<bool>("--bias").value_or(true);
            auto norm_type = args_.get_optional<std::string>("--norm_type").value_or("layer_norm");
            auto hidden_states = args_.get_one<Tensor>("--hidden_states", {ctx, inputs_});
            auto conditioning_embedding = args_.get_one<Tensor>("--conditioning_embedding", {ctx, inputs_});

            AdaLayerNormContinuous model(embedding_dim, conditioning_embedding_dim, elementwise_affine, eps, bias, norm_type.c_str());

            CreateParametersVisitor visitor(ctx, inputs_, args_);
            model.accept(visitor);

            return model.forward(*ctx, hidden_states, conditioning_embedding);
        }

        throw std::runtime_error("Uknown command: " + args_.get(0));
    }

private:
    class CreateParametersVisitor : public Visitor {
    public:
        CreateParametersVisitor(Context& ctx, std::vector<std::pair<Tensor, std::vector<float>>>& inputs, ArgumentParser& args)
            : ctx_(ctx), inputs_(inputs), args_(args)
        {}

        virtual void visit(Parameter& parameter, std::vector<std::string> path) {
            auto joined_path = join_path(path);
            auto tensor = args_.get_one<Tensor>(joined_path, {ctx_, inputs_});

            parameter.set(tensor);
        }

    private:
        Context& ctx_;
        std::vector<std::pair<Tensor, std::vector<float>>>& inputs_;
        ArgumentParser& args_;
        
        static std::string join_path(const std::vector<std::string>& path) {
            return std::accumulate(std::begin(path), std::end(path), std::string("--param"), [](const std::string& acc, const std::string& x) {
                return acc + "-" + x;
            });
        }
    };
};

int main(int argc, char** argv) {
    TestNnCLI cli(argc, argv);
    return cli.main();
}
