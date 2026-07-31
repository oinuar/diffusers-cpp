#include "../TestCLI.hpp"
#include "nn/Parameter.hpp"
#include "nn/Visitor.hpp"
#include "nn/RethrowVisitor.hpp"

#include "nn/Linear.hpp"
#include "nn/SiLU.hpp"
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

    virtual Plan build(Runtime& runtime) {

        if (args_.get(0) == "Linear") {
            auto in_features = args_.get_one<int64_t>("--in_features");
            auto out_features = args_.get_one<int64_t>("--out_features");
            auto bias = args_.get_optional<bool>("--bias").value_or(true);
            auto x = args_.get_one<Tensor>("--x", {runtime});

            Linear model(in_features, out_features, bias);

            CreateParametersVisitor create_parameters(runtime, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            return model.forward(runtime, x);
        }
        
        if (args_.get(0) == "SiLU") {
            auto x = args_.get_one<Tensor>("--x", {runtime});

            SiLU model;

            return model.forward(runtime, x);
        }

        if (args_.get(0) == "RMSNorm") {
            auto dim = args_.get_one<int64_t>("--dim");
            auto eps = args_.get_optional<float>("--eps").value_or(1e-5f);
            auto elementwise_affine = args_.get_optional<bool>("--elementwise_affine").value_or(true);
            auto x = args_.get_one<Tensor>("--x", {runtime});

            RMSNorm model(dim, eps, elementwise_affine);

            CreateParametersVisitor create_parameters(runtime, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            return model.forward(runtime, x);
        }

        if (args_.get(0) == "LayerNorm") {
            auto dim = args_.get_one<int64_t>("--dim");
            auto eps = args_.get_optional<float>("--eps").value_or(1e-5f);
            auto elementwise_affine = args_.get_optional<bool>("--elementwise_affine").value_or(true);
            auto bias = args_.get_optional<bool>("--bias").value_or(true);
            auto x = args_.get_one<Tensor>("--x", {runtime});

            LayerNorm model(dim, eps, elementwise_affine, bias);

            CreateParametersVisitor create_parameters(runtime, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            return model.forward(runtime, x);
        }

        if (args_.get(0) == "GroupNorm") {
            auto num_groups = args_.get_one<int64_t>("--num_groups");
            auto num_channels = args_.get_one<int64_t>("--num_channels");
            auto eps = args_.get_optional<float>("--eps").value_or(1e-5f);
            auto affine = args_.get_optional<bool>("--affine").value_or(true);
            auto bias = args_.get_optional<bool>("--bias").value_or(true);
            auto input = args_.get_one<Tensor>("--input", {runtime});

            GroupNorm model(num_groups, num_channels, eps, affine, bias);

            CreateParametersVisitor create_parameters(runtime, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            return model.forward(runtime, input);
        }

        if (args_.get(0) == "Conv2d") {
            auto in_channels = args_.get_one<int64_t>("--in_channels");
            auto out_channels = args_.get_one<int64_t>("--out_channels");
            auto kernel_size = args_.get_one<int64_t>("--kernel_size");
            auto stride = args_.get_optional<int64_t>("--stride").value_or(1);
            auto padding = args_.get_optional<int64_t>("--padding").value_or(0);
            auto bias = args_.get_optional<bool>("--bias").value_or(true);
            auto x = args_.get_one<Tensor>("--x", {runtime});

            Conv2d model(in_channels, out_channels, kernel_size, stride, padding, bias);

            CreateParametersVisitor create_parameters(runtime, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            return model.forward(runtime, x);
        }

        if (args_.get(0) == "FlashAttention") {
            auto q = args_.get_one<Tensor>("--q", {runtime});
            auto k = args_.get_one<Tensor>("--k", {runtime});
            auto v = args_.get_one<Tensor>("--v", {runtime});
            auto mask = args_.get_optional<Tensor>("--mask", {runtime});

            FlashAttentionOp attention;

            return attention(runtime, q, k, v, mask);
        }

        throw std::runtime_error("Uknown command: " + args_.get(0));
    }

protected:
    class CreateParametersVisitor : public Visitor {
    public:
        CreateParametersVisitor(Runtime& runtime, ArgumentParser& args)
            : runtime_(runtime), args_(args)
        {}

        virtual void visit(Parameter& parameter, std::vector<std::string> path) {
            auto joined_path = join_path(path);
            auto tensor = args_.get_one<Tensor>(joined_path, {runtime_});
            parameter.set(tensor);
        }

    private:
        Runtime& runtime_;
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
