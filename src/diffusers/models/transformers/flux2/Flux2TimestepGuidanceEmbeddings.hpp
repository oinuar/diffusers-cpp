#pragma once

#include "nn/Module.hpp"
#include "diffusers/models/embeddings/TimestepEmbedding.hpp"
#include "diffusers/models/embeddings/Timesteps.hpp"

class Flux2TimestepGuidanceEmbeddings : public Module {
public:
    Flux2TimestepGuidanceEmbeddings(
        int64_t in_channels = 256,
        int64_t embedding_dim = 6144,
        bool bias = false,
        bool guidance_embeds = true
    ) : guidance_embeds_(guidance_embeds) {
        modules["time_proj"] = std::make_shared<Timesteps>(in_channels, true, 0.0f);
        modules["timestep_embedder"] = std::make_shared<TimestepEmbedding<>>(in_channels, embedding_dim, std::nullopt, std::nullopt, bias);

        if (guidance_embeds)
            modules["guidance_embedder"] = std::make_shared<TimestepEmbedding<>>(in_channels, embedding_dim, std::nullopt, std::nullopt, bias);
    }

    Tensor forward(Runtime& runtime, Tensor timestep, std::optional<Tensor> guidance = std::nullopt) {
        auto time_proj = std::static_pointer_cast<Timesteps>(modules["time_proj"]);
        auto timestep_embedder = std::static_pointer_cast<TimestepEmbedding<>>(modules["timestep_embedder"]);

        auto timesteps_proj = time_proj->forward(runtime, timestep);
        auto timesteps_emb = timestep_embedder->forward(runtime, timesteps_proj).to(timestep.dtype());

        if (guidance_embeds_) {
            auto guidance_embedder = std::static_pointer_cast<TimestepEmbedding<>>(modules["guidance_embedder"]);

            auto guidance_proj = time_proj->forward(runtime, guidance.value());
            auto guidance_emb = guidance_embedder->forward(runtime, guidance_proj).to(guidance.value().dtype());

            timesteps_emb = timesteps_emb + guidance_emb;
        }

        return timesteps_emb;
    }

private:
    bool guidance_embeds_;
};
