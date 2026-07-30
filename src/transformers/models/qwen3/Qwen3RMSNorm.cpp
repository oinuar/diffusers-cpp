#include "transformers/models/qwen3/Qwen3RMSNorm.hpp"
#include "nn/Parameter.hpp"

Qwen3RMSNorm::Qwen3RMSNorm(int64_t hidden_size, float eps) : eps(eps) {
    modules["weight"] = std::make_shared<Parameter>(Tensor::Shape({hidden_size}));
}

Tensor Qwen3RMSNorm::forward(Runtime& runtime, Tensor hidden_states) {
    auto weight = std::static_pointer_cast<Parameter>(modules["weight"])->forward();
    auto input_dtype = hidden_states.dtype();
    
    // Preserve exact Python execution order and type casting
    auto hidden_states_f32 = hidden_states.to(GGML_TYPE_F32);
    auto variance = (hidden_states_f32 * hidden_states_f32).mean(-1, true);
    auto rsqrt_var = rsqrt(variance + eps);
    auto normalized = hidden_states_f32 * rsqrt_var;
    
    return (weight * normalized).to(input_dtype);
}
