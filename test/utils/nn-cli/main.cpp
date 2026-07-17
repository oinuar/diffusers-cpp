#include "../TestCLI.hpp"
#include "nn/Parameter.hpp"
#include "nn/Visitor.hpp"
#include "nn/Linear.hpp"
#include "nn/SiLU.hpp"
#include "nn/RMSNorm.hpp"
#include "nn/LayerNorm.hpp"
#include "nn/AdaLayerNormContinuous.hpp"
#include "nn/TimestepEmbedding.hpp"
#include "nn/Timesteps.hpp"

#include "models/diffusers/transformers/flux2/Flux2SwiGLU.hpp"
#include "models/diffusers/transformers/flux2/Flux2FeedForward.hpp"
#include "models/diffusers/transformers/flux2/Flux2Modulation.hpp"
#include "models/diffusers/transformers/flux2/Flux2TimestepGuidanceEmbeddings.hpp"
#include "models/diffusers/transformers/flux2/Flux2PosEmbed.hpp"
#include "models/diffusers/transformers/flux2/Flux2Attention.hpp"
#include "models/attention/FlashAttentionOp.hpp"


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
            auto hidden_states = args_.get_one<Tensor>("--hidden_states", {ctx, inputs_});
            auto conditioning_embedding = args_.get_one<Tensor>("--conditioning_embedding", {ctx, inputs_});

            AdaLayerNormContinuous<> model(embedding_dim, conditioning_embedding_dim, elementwise_affine, eps, bias);

            CreateParametersVisitor visitor(ctx, inputs_, args_);
            model.accept(visitor);

            return model.forward(*ctx, hidden_states, conditioning_embedding);
        }

        if (args_.get(0) == "TimestepEmbedding") {
            auto in_channels = args_.get_one<int64_t>("--in_channels");
            auto time_embed_dim = args_.get_one<int64_t>("--time_embed_dim");
            auto out_dim = args_.get_optional<int64_t>("--out_dim");
            auto cond_proj_dim = args_.get_optional<int64_t>("--cond_proj_dim");
            auto sample_proj_bias = args_.get_optional<bool>("--sample_proj_bias").value_or(true);
            auto sample = args_.get_one<Tensor>("--sample", {ctx, inputs_});
            auto condition = args_.get_optional<Tensor>("--condition", {ctx, inputs_});

            TimestepEmbedding<> model(in_channels, time_embed_dim, out_dim, cond_proj_dim, sample_proj_bias);

            CreateParametersVisitor visitor(ctx, inputs_, args_);
            model.accept(visitor);

            return model.forward(*ctx, sample, condition);
        }

        if (args_.get(0) == "Timesteps") {
            auto num_channels = args_.get_one<int64_t>("--num_channels");
            auto flip_sin_to_cos = args_.get_one<bool>("--flip_sin_to_cos");
            auto downscale_freq_shift = args_.get_one<float>("--downscale_freq_shift");
            auto scale = args_.get_optional<float>("--scale").value_or(1.0);
            auto timesteps = args_.get_one<Tensor>("--timesteps", {ctx, inputs_});

            Timesteps model(num_channels, flip_sin_to_cos, downscale_freq_shift, scale);

            CreateParametersVisitor visitor(ctx, inputs_, args_);
            model.accept(visitor);

            return model.forward(*ctx, timesteps);
        }

        
        if (args_.get(0) == "Flux2SwiGLU") {
            auto x = args_.get_one<Tensor>("--x", {ctx, inputs_});

            Flux2SwiGLU model;

            return model.forward(*ctx, x);
        }

        if (args_.get(0) == "Flux2FeedForward") {
            auto dim = args_.get_one<int64_t>("--dim");
            auto dim_out = args_.get_optional<int64_t>("--dim_out");
            auto mult = args_.get_optional<float>("--mult").value_or(3.0);
            auto inner_dim = args_.get_optional<int64_t>("--inner_dim");
            auto bias = args_.get_optional<bool>("--bias").value_or(false);
            auto x = args_.get_one<Tensor>("--x", {ctx, inputs_});

            Flux2FeedForward model(dim, dim_out, mult, inner_dim, bias);

            CreateParametersVisitor visitor(ctx, inputs_, args_);
            model.accept(visitor);

            return model.forward(*ctx, x);
        }

        if (args_.get(0) == "Flux2Modulation") {
            auto dim = args_.get_one<int64_t>("--dim");
            auto mod_param_sets = args_.get_optional<int64_t>("--mod_param_sets").value_or(2);
            auto bias = args_.get_optional<bool>("--bias").value_or(false);
            auto temb = args_.get_one<Tensor>("--temb", {ctx, inputs_});

            Flux2Modulation model(dim, mod_param_sets, bias);

            CreateParametersVisitor visitor(ctx, inputs_, args_);
            model.accept(visitor);

            return model.forward(*ctx, temb);
        }

        if (args_.get(0) == "Flux2TimestepGuidanceEmbeddings") {
            auto in_channels = args_.get_one<int64_t>("--in_channels");
            auto embedding_dim = args_.get_one<int64_t>("--embedding_dim");
            auto bias = args_.get_optional<bool>("--bias").value_or(false);
            auto guidance_embeds = args_.get_optional<bool>("--guidance_embeds").value_or(true);
            auto timestep = args_.get_one<Tensor>("--timestep", {ctx, inputs_});
            auto guidance = args_.get_optional<Tensor>("--guidance", {ctx, inputs_});

            Flux2TimestepGuidanceEmbeddings model(in_channels, embedding_dim, bias, guidance_embeds);

            CreateParametersVisitor visitor(ctx, inputs_, args_);
            model.accept(visitor);

            return model.forward(*ctx, timestep, guidance);
        }

        if (args_.get(0) == "Flux2PosEmbed") {
            auto theta = args_.get_one<int64_t>("--theta");
            auto axes_dim = args_.get_many<int64_t>("--axes_dim");
            auto ids = args_.get_one<Tensor>("--ids", {ctx, inputs_});

            Flux2PosEmbed model(theta, axes_dim);

            CreateParametersVisitor visitor(ctx, inputs_, args_);
            model.accept(visitor);

            auto result = model.forward(*ctx, ids);

            return {{result.first, result.second}};
        }


        if (args_.get(0) == "Flux2Attention") {
            auto query_dim = args_.get_one<int64_t>("--query_dim");
            auto heads = args_.get_optional<int64_t>("--heads").value_or(8);
            auto dim_head = args_.get_optional<int64_t>("--dim_head").value_or(64);
            auto dropout = args_.get_optional<float>("--dropout").value_or(0.0);
            auto bias = args_.get_optional<bool>("--bias").value_or(false);
            auto added_kv_proj_dim = args_.get_optional<int64_t>("--added_kv_proj_dim");
            auto added_proj_bias = args_.get_optional<bool>("--added_proj_bias").value_or(true);
            auto out_bias = args_.get_optional<bool>("--out_bias").value_or(true);
            auto eps = args_.get_optional<float>("--eps").value_or(1e-5);
            auto out_dim = args_.get_optional<int64_t>("--out_dim");
            auto elementwise_affine = args_.get_optional<bool>("--elementwise_affine").value_or(true);
            auto hidden_states = args_.get_one<Tensor>("--hidden_states", {ctx, inputs_});
            auto encoder_hidden_states = args_.get_optional<Tensor>("--encoder_hidden_states", {ctx, inputs_});
            auto attention_mask = args_.get_optional<Tensor>("--attention_mask", {ctx, inputs_});
            auto image_rotary_emb_0 = args_.get_optional<Tensor>("--image_rotary_emb-0", {ctx, inputs_});
            auto image_rotary_emb_1 = args_.get_optional<Tensor>("--image_rotary_emb-1", {ctx, inputs_});
            auto image_rotary_emb = image_rotary_emb_0 && image_rotary_emb_1 ? std::make_optional(std::make_tuple(
                image_rotary_emb_0.value(),
                image_rotary_emb_1.value()
            )) : std::nullopt;

            Flux2Attention<FlashAttentionOp> model(
                query_dim,
                heads,
                dim_head,
                dropout,
                bias,
                added_kv_proj_dim,
                added_proj_bias,
                out_bias,
                eps,
                out_dim,
                elementwise_affine
            );

            CreateParametersVisitor visitor(ctx, inputs_, args_);
            model.accept(visitor);

            auto [y1, y2] = model.forward(*ctx, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb);
            std::vector<Tensor> results;

            results.push_back(y1);

            if (y2)
                results.push_back(y2.value());

            return results;
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
