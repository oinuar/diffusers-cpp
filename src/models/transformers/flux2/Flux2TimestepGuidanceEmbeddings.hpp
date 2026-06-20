#pragma once

#include "modules/Module.hpp"
#include "models/embeddings/Timesteps.hpp"
#include "models/embeddings/TimestepEmbedding.hpp"

class Flux2TimestepGuidanceEmbeddings : public Module {
public:
    Flux2TimestepGuidanceEmbeddings(
        int64_t in_channels = 256,
        int64_t embedding_dim = 6144,
        bool bias = false,
        bool guidance_embeds = true
    ) : guidance_embeds_(guidance_embeds) {
        modules["time_proj"] = std::make_shared<Timesteps>(in_channels, true, 0.0f);
        modules["timestep_embedder"] = std::make_shared<TimestepEmbedding<>>(in_channels, embedding_dim, bias);

        if (guidance_embeds)
            modules["guidance_embedder"] = std::make_shared<TimestepEmbedding<>>(in_channels, embedding_dim, bias);
    }

    Tensor forward(ggml_context* ctx, Tensor timestep, std::optional<Tensor> guidance = std::nullopt) {
        auto time_proj = std::static_pointer_cast<Timesteps>(modules["time_proj"]);
        auto timestep_embedder = std::static_pointer_cast<TimestepEmbedding<>>(modules["timestep_embedder"]);

        auto timesteps_proj = time_proj->forward(ctx, timestep);
        auto timesteps_emb = timestep_embedder->forward(ctx, timesteps_proj);

        if (guidance_embeds_) {
            auto guidance_embedder = std::static_pointer_cast<TimestepEmbedding<>>(modules["guidance_embedder"]);

            auto guidance_proj = time_proj->forward(ctx, guidance.value());
            auto guidance_emb = guidance_embedder->forward(ctx, guidance_proj);

            timesteps_emb = timesteps_emb + guidance_emb;
        }

        return timesteps_emb;
    }

private:
    bool guidance_embeds_;
};
