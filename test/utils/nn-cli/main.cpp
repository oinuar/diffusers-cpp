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

    virtual std::vector<Tensor> compute(Scheduler& scheduler, Context& context, Allocator& allocator, std::optional<Context>& local_context, std::optional<DeviceAllocator>& local_allocator) {
        if (args_.get(0) == "Linear") {
            auto in_features = args_.get_one<int64_t>("--in_features");
            auto out_features = args_.get_one<int64_t>("--out_features");
            auto bias = args_.get_optional<bool>("--bias").value_or(true);
            auto x = args_.get_one<Tensor>("--x", {local_context ? *local_context : context});

            Linear model(in_features, out_features, bias);

            // These are parameters, they are OK and should go to this context
            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            // TODO: allocation cannot happen here, since we need to Plan forward(). 
            // Then we can allocate once we know the splits.
            if (local_allocator)
                allocator.allocate(GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

            // TODO: This should be called twice with two scopes: Plan and Execution
            auto output = model.forward(local_context ? *local_context : context, x);

            Graph graph(scheduler, local_context ? *local_context : context, {output});

            if (local_allocator)
                 local_allocator->allocate();

            Computation computation(graph, {&context, local_context ? &(*local_context) : nullptr});
            return computation().results();
        }
        
        if (args_.get(0) == "SiLU") {
            auto x = args_.get_one<Tensor>("--x", {local_context ? *local_context : context});

            SiLU model;

            auto output = model.forward(local_context ? *local_context : context, x);

            Graph graph(scheduler, local_context ? *local_context : context, {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, local_context ? &(*local_context) : nullptr});
            return computation().results();
        }

        if (args_.get(0) == "RMSNorm") {
            auto dim = args_.get_one<int64_t>("--dim");
            auto eps = args_.get_optional<float>("--eps").value_or(1e-5f);
            auto elementwise_affine = args_.get_optional<bool>("--elementwise_affine").value_or(true);
            auto x = args_.get_one<Tensor>("--x", {local_context ? *local_context : context});

            RMSNorm model(dim, eps, elementwise_affine);

            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            if (local_allocator)
                allocator.allocate(GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

            auto output = model.forward(local_context ? *local_context : context, x);

            Graph graph(scheduler, local_context ? *local_context : context, {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, local_context ? &(*local_context) : nullptr});
            return computation().results();
        }

        if (args_.get(0) == "LayerNorm") {
            auto dim = args_.get_one<int64_t>("--dim");
            auto eps = args_.get_optional<float>("--eps").value_or(1e-5f);
            auto elementwise_affine = args_.get_optional<bool>("--elementwise_affine").value_or(true);
            auto bias = args_.get_optional<bool>("--bias").value_or(true);
            auto x = args_.get_one<Tensor>("--x", {local_context ? *local_context : context});

            LayerNorm model(dim, eps, elementwise_affine, bias);

            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            if (local_allocator)
                allocator.allocate(GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

            auto output = model.forward(local_context ? *local_context : context, x);

            Graph graph(scheduler, local_context ? *local_context : context, {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, local_context ? &(*local_context) : nullptr});
            return computation().results();
        }

        if (args_.get(0) == "GroupNorm") {
            auto num_groups = args_.get_one<int64_t>("--num_groups");
            auto num_channels = args_.get_one<int64_t>("--num_channels");
            auto eps = args_.get_optional<float>("--eps").value_or(1e-5f);
            auto affine = args_.get_optional<bool>("--affine").value_or(true);
            auto bias = args_.get_optional<bool>("--bias").value_or(true);
            auto input = args_.get_one<Tensor>("--input", {local_context ? *local_context : context});

            GroupNorm model(num_groups, num_channels, eps, affine, bias);

            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            if (local_allocator)
                allocator.allocate(GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

            auto output = model.forward(local_context ? *local_context : context, input);

            Graph graph(scheduler, local_context ? *local_context : context, {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, local_context ? &(*local_context) : nullptr});
            return computation().results();
        }

        if (args_.get(0) == "Conv2d") {
            auto in_channels = args_.get_one<int64_t>("--in_channels");
            auto out_channels = args_.get_one<int64_t>("--out_channels");
            auto kernel_size = args_.get_one<int64_t>("--kernel_size");
            auto stride = args_.get_optional<int64_t>("--stride").value_or(1);
            auto padding = args_.get_optional<int64_t>("--padding").value_or(0);
            auto bias = args_.get_optional<bool>("--bias").value_or(true);
            auto x = args_.get_one<Tensor>("--x", {local_context ? *local_context : context});

            Conv2d model(in_channels, out_channels, kernel_size, stride, padding, bias);

            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            if (local_allocator)
                allocator.allocate(GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

            auto output = model.forward(local_context ? *local_context : context, x);

            Graph graph(scheduler, local_context ? *local_context : context, {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, local_context ? &(*local_context) : nullptr});
            return computation().results();
        }

        if (args_.get(0) == "FlashAttention") {
            auto q = args_.get_one<Tensor>("--q", {local_context ? *local_context : context});
            auto k = args_.get_one<Tensor>("--k", {local_context ? *local_context : context});
            auto v = args_.get_one<Tensor>("--v", {local_context ? *local_context : context});
            auto mask = args_.get_optional<Tensor>("--mask", {local_context ? *local_context : context});

            FlashAttentionOp attention;

            auto output = attention(context, q, k, v, mask);

            Graph graph(scheduler, local_context ? *local_context : context, {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, local_context ? &(*local_context) : nullptr});
            return computation().results();
        }

        if (args_.get(0) == "Embedding") {
            auto num_embeddings = args_.get_one<int64_t>("--num_embeddings");
            auto embedding_dim = args_.get_one<int64_t>("--embedding_dim");
            auto padding_idx = args_.get_optional<int64_t>("--padding_idx");
            auto input = args_.get_one<Tensor>("--input", {local_context ? *local_context : context});

            Embedding model(num_embeddings, embedding_dim, padding_idx);

            CreateParametersVisitor create_parameters(context, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            if (local_allocator)
                allocator.allocate(GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

            auto output = model.forward(local_context ? *local_context : context, input);

            Graph graph(scheduler, local_context ? *local_context : context, {output});

            if (local_allocator)
                local_allocator->allocate();

            Computation computation(graph, {&context, local_context ? &(*local_context) : nullptr});
            return computation().results();
        }

        throw std::runtime_error("Uknown command: " + args_.get(0));
    }
};

int main(int argc, char** argv) {
    TestNnCLI cli(argc, argv);
    return cli.main();
}
