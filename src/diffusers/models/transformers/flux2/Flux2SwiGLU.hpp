#pragma once

#include "nn/Module.hpp"
#include "nn/SiLU.hpp"

class Flux2SwiGLU : public Module {
public:
    Flux2SwiGLU() {
        modules["gate_fn"] = std::make_shared<SiLU>();
    }

    // The gate and up projections are separate tensors: the reference
    // chunks the fused (gate | up) output along its last dim, but a
    // windowed view of a sharded tensor cannot carry the shard on the
    // meta device, so the feedforward computes the two projections
    // separately and passes the halves in.
    Tensor forward(Scope scope, Tensor gate, Tensor up) {
        auto gate_fn = std::static_pointer_cast<SiLU>(modules["gate_fn"]);

        return gate_fn->forward(scope, gate) * up;
    }
};
