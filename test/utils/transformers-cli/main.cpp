#include "../TestCLI.hpp"
#include "nn/Parameter.hpp"
#include "nn/Visitor.hpp"
#include "nn/RethrowVisitor.hpp"

#include "transformers/models/qwen2/Qwen2TokenizerFast.hpp"

class TestTransformersCLI : public TestCLI {
public:
    TestTransformersCLI(int argc, char** argv) : TestCLI(argc, argv) {}

    virtual Plan build(Runtime& runtime) {
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
