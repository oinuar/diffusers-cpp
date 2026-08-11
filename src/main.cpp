#include "ggml/Backend.hpp"
#include "ggml/Runtime.hpp"
#include "ggml/Scheduler.hpp"
#include "ggml/Computation.hpp"
#include "ggml/GGUFLoaderVisitor.hpp"
#include "nn/RethrowVisitor.hpp"
#include "diffusers/models/transformers/flux2/Flux2Transformer2DModel.hpp"
#include "diffusers/models/autoencoders/AutoencoderKLFlux2.hpp"
#include "transformers/models/qwen3/Qwen3Config.hpp"
#include "transformers/models/qwen3/Qwen3ForCausalLM.hpp"
#include "transformers/models/qwen2/Qwen2TokenizerFast.hpp"
#include <iostream>
#include <filesystem>

int main() {
    Qwen2TokenizerFast tokenizer("../utils/convert-model/tokenizer/tokenizer.json");

    auto text = tokenizer.apply_chat_template({{
        "user", "hello world"
    }});

    std::cerr << "template: " << text <<std::endl;

    auto tokens = tokenizer.encode(text);

    for (auto& token : tokens)
        std::cerr << "token: " << token << std::endl;

    return 0;
}
