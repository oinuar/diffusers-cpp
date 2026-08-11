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
#include "transformers/models/qwen3/Qwen3DecoderLayer.hpp"
#include "transformers/models/qwen3/Qwen3Model.hpp"
#include "transformers/models/qwen3/Qwen3ForCausalLM.hpp"

class TestQwen3CLI : public TestCLI {
public:
    TestQwen3CLI(int argc, char** argv) : TestCLI(argc, argv) {}

    virtual std::vector<Tensor> compute(Runtime& runtime) {
        if (args_.get(0) == "Qwen3RMSNorm") {
            auto hidden_size = args_.get_one<int64_t>("--hidden_size");
            auto hidden_states = args_.get_one<Tensor>("--hidden_states", {runtime});

            Qwen3RMSNorm model(hidden_size);

            CreateParametersVisitor create_parameters(runtime, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto output = model.forward(runtime, hidden_states);

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "Qwen3MLP") {
            Qwen3Config config;
            config.hidden_size = args_.get_optional<int64_t>("--hidden_size").value_or(config.hidden_size);
            config.intermediate_size = args_.get_optional<int64_t>("--intermediate_size").value_or(config.intermediate_size);

            auto hidden_states = args_.get_one<Tensor>("--hidden_states", {runtime});

            Qwen3MLP model(config);

            CreateParametersVisitor create_parameters(runtime, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto output = model.forward(runtime, hidden_states);

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "Qwen3RotaryEmbedding") {
            Qwen3Config config;
            config.head_dim = args_.get_optional<int64_t>("--head_dim").value_or(config.head_dim);
            config.rope_theta = args_.get_optional<int64_t>("--rope_theta").value_or(config.rope_theta);

            auto x = args_.get_one<Tensor>("--x", {runtime});
            auto position_ids = args_.get_one<Tensor>("--position_ids", {runtime});

            Qwen3RotaryEmbedding model(config);

            auto output = model.forward(runtime, x, position_ids);

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
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

            Qwen3Attention<ScaledDotProductAttention<FlashAttentionOp>> model(config, layer_idx);
            Qwen3RotaryEmbedding rotary_emb(config);

            CreateParametersVisitor create_parameters(runtime, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto output = model.forward(runtime, rotary_emb, hidden_states, position_ids, attention_mask, past_key_values);

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
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

            Qwen3Attention<ScaledDotProductAttention<FlashAttentionOp>> model(config, layer_idx);
            Qwen3RotaryEmbedding rotary_emb(config);

            CreateParametersVisitor create_parameters(runtime, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto output = model.forward(runtime, rotary_emb, hidden_states, position_ids, attention_mask, past_key_values);

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "Qwen3DecoderLayer") {
            Qwen3Config config;
            config.hidden_size = args_.get_optional<int64_t>("--hidden_size").value_or(config.hidden_size);
            config.intermediate_size = args_.get_optional<int64_t>("--intermediate_size").value_or(config.intermediate_size);
            config.num_attention_heads = args_.get_optional<int64_t>("--num_attention_heads").value_or(config.num_attention_heads);
            config.num_key_value_heads = args_.get_optional<int64_t>("--num_key_value_heads").value_or(config.num_key_value_heads);
            config.max_position_embeddings = args_.get_optional<int64_t>("--num_key_value_heads").value_or(config.max_position_embeddings);

            auto layer_idx = args_.get_one<int>("--layer_idx");
            auto hidden_states = args_.get_one<Tensor>("--hidden_states", {runtime});
            auto position_ids = args_.get_one<Tensor>("--position_ids", {runtime});

            Qwen3DecoderLayer model(config, layer_idx);
            Qwen3RotaryEmbedding rotary_emb(config);

            CreateParametersVisitor create_parameters(runtime, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto output = model.forward(runtime, rotary_emb, hidden_states, position_ids);

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "Qwen3Model") {
            Qwen3Config config;
            config.vocab_size = args_.get_optional<int64_t>("--vocab_size").value_or(config.vocab_size);
            config.hidden_size = args_.get_optional<int64_t>("--hidden_size").value_or(config.hidden_size);
            config.intermediate_size = args_.get_optional<int64_t>("--intermediate_size").value_or(config.intermediate_size);
            config.num_hidden_layers = args_.get_optional<int64_t>("--num_hidden_layers").value_or(config.num_hidden_layers);
            config.num_attention_heads = args_.get_optional<int64_t>("--num_attention_heads").value_or(config.num_attention_heads);
            config.num_key_value_heads = args_.get_optional<int64_t>("--num_key_value_heads").value_or(config.num_key_value_heads);
            config.attention_bias = args_.get_optional<int64_t>("--attention_bias").value_or(config.attention_bias);
            config.rope_theta = args_.get_optional<int64_t>("--rope_theta").value_or(config.rope_theta);
            config.max_position_embeddings = args_.get_optional<int64_t>("--max_position_embeddings").value_or(config.max_position_embeddings);
            config.pad_token_id = args_.get_optional<int64_t>("--pad_token_id");
            config.head_dim = args_.get_optional<int64_t>("--head_dim").value_or(config.head_dim);

            auto input_ids = args_.get_optional<Tensor>("--input_ids", {runtime});
            auto input_embeds = args_.get_optional<Tensor>("--input_embeds", {runtime});
            auto attention_mask = args_.get_optional<Tensor>("--attention_mask", {runtime});
            auto position_ids = args_.get_optional<Tensor>("--position_ids", {runtime});
            auto past_key_values = args_.get_optional<Tensor>("--past_key_values", {runtime});
            auto use_cache = args_.get_optional<bool>("--past_key_values");
            auto output_hidden_states = args_.get_optional<bool>("--output_hidden_states").value_or(false);

            Qwen3Model model(config);

            CreateParametersVisitor create_parameters(runtime, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            std::vector<Tensor> hidden_states;

            auto output = model.forward(
                runtime,
                input_ids,
                input_embeds,
                attention_mask,
                position_ids,
                past_key_values,
                use_cache,
                output_hidden_states ? &hidden_states : nullptr);

            if (output_hidden_states) {
                Graph graph(runtime, std::move(hidden_states));
                Computation computation(graph);
                return computation.results();
            }
            
            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
        }

        if (args_.get(0) == "Qwen3ForCausalLM") {
            Qwen3Config config;
            config.vocab_size = args_.get_optional<int64_t>("--vocab_size").value_or(config.vocab_size);
            config.hidden_size = args_.get_optional<int64_t>("--hidden_size").value_or(config.hidden_size);
            config.intermediate_size = args_.get_optional<int64_t>("--intermediate_size").value_or(config.intermediate_size);
            config.num_hidden_layers = args_.get_optional<int64_t>("--num_hidden_layers").value_or(config.num_hidden_layers);
            config.num_attention_heads = args_.get_optional<int64_t>("--num_attention_heads").value_or(config.num_attention_heads);
            config.num_key_value_heads = args_.get_optional<int64_t>("--num_key_value_heads").value_or(config.num_key_value_heads);
            config.max_position_embeddings = args_.get_optional<int64_t>("--max_position_embeddings").value_or(config.max_position_embeddings);

            auto input_ids = args_.get_optional<Tensor>("--input_ids", {runtime});
            auto attention_mask = args_.get_optional<Tensor>("--attention_mask", {runtime});
            auto position_ids = args_.get_optional<Tensor>("--position_ids", {runtime});
            auto past_key_values = args_.get_optional<Tensor>("--past_key_values", {runtime});
            auto inputs_embeds = args_.get_optional<Tensor>("--inputs_embeds", {runtime});
            auto labels = args_.get_optional<Tensor>("--labels", {runtime});
            auto use_cache = args_.get_optional<bool>("--use_cache");
            auto logits_to_keep = args_.get_optional<int>("--logits_to_keep").value_or(0);
            
            Qwen3ForCausalLM model(config);

            CreateParametersVisitor create_parameters(runtime, args_);
            RethrowVisitor visitor(create_parameters);
            model.accept(visitor);
            visitor.rethrow();

            auto output = model.forward(
                runtime,
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
                use_cache,
                logits_to_keep
            );

            Graph graph(runtime, {output});
            Computation computation(graph);
            return computation.results();
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
    TestQwen3CLI cli(argc, argv);

    if (cli.args().get(0).rfind("Qwen2TokenizerFast", 0) == 0) {
        auto tokenizer_file = cli.args().get_one<std::string>("--tokenizer_file");

        Qwen2TokenizerFast tokenizer(tokenizer_file);

        if (cli.args().get(0) == "Qwen2TokenizerFast_Encode") {
            auto text = cli.args().get_one<std::string>("--text");
            auto max_length = cli.args().get_optional<int>("--max_length").value_or(0);
            auto add_special_tokens = cli.args().get_optional<bool>("--add_special_tokens").value_or(true);
            auto return_attention_mask = cli.args().get_optional<bool>("--return_attention_mask").value_or(false);
            
            std::vector<int> mask;

            auto tokens = tokenizer.encode(text, max_length, add_special_tokens, return_attention_mask ? &mask : nullptr);

            cli.print_tensor_like(tokens, Tensor::Shape{(int64_t)tokens.size()});

            if (return_attention_mask)
                cli.print_tensor_like(mask, Tensor::Shape{(int64_t)mask.size()});

            return EXIT_SUCCESS;
        }

        if (cli.args().get(0) == "Qwen2TokenizerFast_Decode") {
            auto ids = cli.args().get_many<int>("--ids");
            auto skip_special_tokens = cli.args().get_optional<bool>("--skip_special_tokens").value_or(true);

            auto text = tokenizer.decode(ids, skip_special_tokens);

            cli.print_tensor_like<std::string>({text}, Tensor::Shape{1});
            return EXIT_SUCCESS;
        }

        if (cli.args().get(0) == "Qwen2TokenizerFast_ApplyChatTemplate") {
            auto add_generation_prompt = cli.args().get_optional<bool>("--add_generation_prompt").value_or(true);
            auto enable_thinking = cli.args().get_optional<bool>("--enable_thinking").value_or(false);
            auto json_messages = cli.args().get_many<nlohmann::json>("--messages");

            std::vector<Qwen2TokenizerFast::Message> messages;
            messages.reserve(json_messages.size());

            for (auto& json : json_messages)
                messages.push_back({json["role"], json["content"]});

            auto text = tokenizer.apply_chat_template(messages, add_generation_prompt, enable_thinking);

            cli.print_tensor_like<std::string>({text}, Tensor::Shape{1});
            return EXIT_SUCCESS;
        }
    }

    return cli.main();
}
