#include "../TestCLI.hpp"
#include "nn/Parameter.hpp"
#include "nn/Visitor.hpp"

#include "nn/Linear.hpp"
#include "nn/SiLU.hpp"
#include "nn/modules/normalization/RMSNorm.hpp"
#include "nn/modules/normalization/LayerNorm.hpp"
#include "nn/attention/ScaledDotProductAttention.hpp"
#include "nn/attention/FlashAttentionOp.hpp"

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
            visitor.rethrow();

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
            visitor.rethrow();

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
            visitor.rethrow();

            return model.forward(*ctx, x);
        }

        if (args_.get(0) == "FlashAttention") {
            auto q = args_.get_one<Tensor>("--q", {ctx, inputs_});
            auto k = args_.get_one<Tensor>("--k", {ctx, inputs_});
            auto v = args_.get_one<Tensor>("--v", {ctx, inputs_});
            auto mask = args_.get_optional<Tensor>("--mask", {ctx, inputs_});

            FlashAttentionOp attention;

            return attention(*ctx, q, k, v, mask);
        }

        throw std::runtime_error("Uknown command: " + args_.get(0));
    }

protected:
    class CreateParametersVisitor : public Visitor {
    public:
        CreateParametersVisitor(Context& ctx, std::vector<std::pair<Tensor, std::vector<float>>>& inputs, ArgumentParser& args)
            : ctx_(ctx), inputs_(inputs), args_(args)
        {}

        virtual void visit(Parameter& parameter, std::vector<std::string> path) {
            auto joined_path = join_path(path);

            try {
                auto tensor = args_.get_one<Tensor>(joined_path, {ctx_, inputs_});
                parameter.set(tensor);
            } catch (const std::runtime_error& error) {
                errors_ += "\n  - ";
                errors_ += error.what();
            }
        }

        void rethrow() {
            if (!errors_.empty())
                throw std::runtime_error("There were following errors while creating parameters: " + errors_);
        }

    private:
        Context& ctx_;
        std::vector<std::pair<Tensor, std::vector<float>>>& inputs_;
        ArgumentParser& args_;
        std::string errors_;
        
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
