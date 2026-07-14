#pragma once

#include "nn/Module.hpp"
#include "nn/SiLU.hpp"

class Flux2SwiGLU : public Module {
public:
    Flux2SwiGLU() {
        modules["gate_fn"] = std::make_shared<SiLU>();
    }

    Tensor forward(ggml_context* ctx, Tensor x) {
        auto gate_fn = std::static_pointer_cast<SiLU>(modules["gate_fn"]);
        auto chunks = x.chunk(2, -1);
        auto x1 = chunks.at(0);
        auto x2 = chunks.at(1).contiguous();

        x = gate_fn->forward(ctx, x1) * x2;

        return x;
    }
};
