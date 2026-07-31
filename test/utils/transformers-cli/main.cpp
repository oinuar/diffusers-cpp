#include "../TestCLI.hpp"
#include "nn/Parameter.hpp"
#include "nn/Visitor.hpp"
#include "nn/RethrowVisitor.hpp"
#include "nn/attention/ScaledDotProductAttention.hpp"
#include "nn/attention/FlashAttentionOp.hpp"

#include "transformers/models/qwen2/Qwen2TokenizerFast.hpp"
#include "transformers/models/qwen3/Qwen3RMSNorm.hpp"
#include "transformers/models/qwen3/Qwen3Config.hpp"
#include "transformers/models/qwen3/Qwen3MLP.hpp"
#include "transformers/models/qwen3/Qwen3RotaryEmbedding.hpp"
#include "transformers/models/qwen3/Qwen3Attention.hpp"

class TestTransformersCLI : public TestCLI {
public:
    TestTransformersCLI(int argc, char** argv) : TestCLI(argc, argv) {}

    virtual Plan build(Runtime& runtime) {
        if (args_.get(0) == "Qwen3RMSNorm") {
            auto hidden_size = args_.get_one<int64_t>("--hidden_size");
            auto hidden_states = args_.get_one<Tensor>("--hidden_states", {runtime});

            Qwen3RMSNorm model(hidden_size);

            CreateParametersVisitor create_parameters(runtime, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            return model.forward(runtime, hidden_states);
        }

        if (args_.get(0) == "Qwen3MLP") {
            auto hidden_size = args_.get_one<int64_t>("--hidden_size");
            auto intermediate_size = args_.get_one<int64_t>("--intermediate_size");
            auto hidden_states = args_.get_one<Tensor>("--hidden_states", {runtime});

            Qwen3Config config;
            config.hidden_size = hidden_size;
            config.intermediate_size = intermediate_size;

            Qwen3MLP model(config);

            CreateParametersVisitor create_parameters(runtime, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            return model.forward(runtime, hidden_states);
        }

        if (args_.get(0) == "Qwen3RotaryEmbedding") {
            auto head_dim = args_.get_one<int64_t>("--head_dim");
            auto rope_theta = args_.get_one<float>("--rope_theta");
            auto x = args_.get_one<Tensor>("--x", {runtime});
            auto position_ids = args_.get_one<Tensor>("--position_ids", {runtime});

            Qwen3Config config;
            config.head_dim = head_dim;
            config.rope_theta = rope_theta;

            Qwen3RotaryEmbedding model(config);

            return model.forward(runtime, x, position_ids);
        }

        if (args_.get(0) == "Qwen3Attention") {
            Qwen3Config config;
            
            config.head_dim = args_.get_optional<int64_t>("--head_dim").value_or(config.head_dim);
            config.hidden_size = args_.get_optional<int64_t>("--hidden_size").value_or(config.hidden_size);
            config.num_attention_heads = args_.get_optional<int64_t>("--num_attention_heads").value_or(config.num_attention_heads);
            config.num_key_value_heads = args_.get_optional<int64_t>("--num_key_value_heads").value_or(config.num_key_value_heads);

            auto position_ids = args_.get_one<Tensor>("--position_ids", {runtime});
            auto hidden_states = args_.get_one<Tensor>("--hidden_states", {runtime});
            auto attention_mask = args_.get_optional<Tensor>("--attention_mask", {runtime});
            auto past_key_values = args_.get_optional<Tensor>("--past_key_values", {runtime});
            auto layer_idx = args_.get_one<int>("--layer_idx");

            Qwen3Attention<FlashAttentionOp> model(config, layer_idx);
            Qwen3RotaryEmbedding rotary_emb(config);

            CreateParametersVisitor create_parameters(runtime, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            return model.forward(runtime, rotary_emb, hidden_states, position_ids, attention_mask, past_key_values);
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
    TestTransformersCLI cli(argc, argv);

    if (cli.args().get(0) == "Qwen2TokenizerFast_Encode") {
        auto tokenizer_file = cli.args().get_one<std::string>("--tokenizer_file");
        auto text = cli.args().get_one<std::string>("--text");
        auto add_special_tokens = cli.args().get_optional<bool>("--add_special_tokens").value_or(true);

        Qwen2TokenizerFast tokenizer(tokenizer_file);

        auto tokens = tokenizer.encode(text, add_special_tokens);

        cli.print_tensor_like(tokens, Tensor::Shape{(int64_t)tokens.size()});
        return EXIT_SUCCESS;
    }

    if (cli.args().get(0) == "Qwen2TokenizerFast_Decode") {
        auto tokenizer_file = cli.args().get_one<std::string>("--tokenizer_file");
        auto ids = cli.args().get_many<int>("--ids");
        auto skip_special_tokens = cli.args().get_optional<bool>("--skip_special_tokens").value_or(true);

        Qwen2TokenizerFast tokenizer(tokenizer_file);

        auto text = tokenizer.decode(ids, skip_special_tokens);

        cli.print_tensor_like<std::string>({text}, Tensor::Shape{1});
        return EXIT_SUCCESS;
    }

    if (cli.args().get(0) == "Qwen2TokenizerFast_ApplyChatTemplate") {
        auto tokenizer_file = cli.args().get_one<std::string>("--tokenizer_file");
        auto add_generation_prompt = cli.args().get_optional<bool>("--add_generation_prompt").value_or(true);
        auto json_messages = cli.args().get_many<nlohmann::json>("--messages");

        std::vector<Qwen2TokenizerFast::Message> messages;
        messages.reserve(json_messages.size());

        for (auto& json : json_messages)
            messages.push_back({json["role"], json["content"]});

        Qwen2TokenizerFast tokenizer(tokenizer_file);

        auto text = tokenizer.apply_chat_template(messages, add_generation_prompt);

        cli.print_tensor_like<std::string>({text}, Tensor::Shape{1});
        return EXIT_SUCCESS;
    }

    return cli.main();
}
