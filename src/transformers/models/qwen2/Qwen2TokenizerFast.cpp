#include "transformers/models/qwen2/Qwen2TokenizerFast.hpp"
#include <fstream>
#include <nlohmann/json.hpp>

using namespace tokenizers;

Qwen2TokenizerFast Qwen2TokenizerFast::from_pretrained(const std::filesystem::path& path) {
    return std::move(Qwen2TokenizerFast(
        path / "tokenizer.json",
        path / "tokenizer_config.json"
    ));
}

Qwen2TokenizerFast::Qwen2TokenizerFast(const std::filesystem::path& tokenizer_file, const std::filesystem::path& config_file)
    : PreTrainedTokenizer(tokenizer_file)
{
    std::ifstream file(config_file, std::ios::binary);
    nlohmann::json j;
    file >> j;

    // 1. Read pad token
    auto pad_token = j["pad_token"].get<std::string>();

    // 2. Resolve pad token id from added tokens
    for (auto& [id, token] : j["added_tokens_decoder"].items()) {
        auto content = token.at("content").get<std::string>();

        if (content == pad_token) {
            pad_token_id_ = std::stoi(id);
            break;
        }
    }
}

int Qwen2TokenizerFast::pad_token_id() const {
    return pad_token_id_;
}

std::string Qwen2TokenizerFast::apply_chat_template(
    const std::vector<Message>& messages,
    bool add_generation_prompt,
    bool enable_thinking,
    const std::vector<Tool>& tools) const
{
    std::string out;

    if (messages.empty()) {
        if (add_generation_prompt) {
            out += "<|im_start|>assistant\n";

            if (!enable_thinking) {
                out += "<think>\n\n</think>\n\n";
            }
        }

        return out;
    }

    // ---------------------------------------------------------------------
    // Jinja:
    //
    // {%- if tools %}
    //     ...
    // {%- else %}
    //     ...
    // {%- endif %}
    // ---------------------------------------------------------------------

    if (!tools.empty()) {
        out += "<|im_start|>system\n";

        // The template only emits the first system message here.
        if (messages[0].role == "system") {
            out += messages[0].content;
            out += "\n\n";
        }

        out += "# Tools\n\n";
        out += "You may call one or more functions to assist with the user query.\n\n";
        out += "You are provided with function signatures within <tools></tools> XML tags:\n";
        out += "<tools>";

        for (const auto& tool : tools) {
            out += "\n";
            out += tool.json;
        }

        out +=
            "\n</tools>\n\n"
            "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
            "</tool_call><|im_end|>\n";
    }
    else {
        // Jinja:
        //
        // {%- if messages[0].role == 'system' %}
        //     {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
        // {%- endif %}

        if (messages[0].role == "system") {
            out += "<|im_start|>system\n";
            out += messages[0].content;
            out += "<|im_end|>\n";
        }
    }

    // ---------------------------------------------------------------------
    // Find last_query_index.
    //
    // Jinja:
    //
    // {% set ns = namespace(multi_step_tool=true,
    //                       last_query_index=messages|length - 1) %}
    //
    // {% for message in messages[::-1] %}
    //     {% set index = (messages|length - 1) - loop.index0 %}
    //
    //     {% if ns.multi_step_tool and
    //           message.role == "user" and
    //           message.content is string and
    //           not(message.content.startswith('<tool_response>')
    //               and message.content.endswith('</tool_response>')) %}
    //
    //         {% set ns.multi_step_tool = false %}
    //         {% set ns.last_query_index = index %}
    //     {% endif %}
    // {% endfor %}
    // ---------------------------------------------------------------------

    size_t last_query_index = messages.size() - 1;
    bool multi_step_tool = true;

    for (size_t reverse = 0; reverse < messages.size(); ++reverse) {
        size_t index = messages.size() - 1 - reverse;

        const Message& message = messages[index];

        if (multi_step_tool &&
            message.role == "user") {

            const std::string& content = message.content;

            bool is_tool_response =
                content.size() >= 15 &&
                content.compare(
                    0,
                    std::string("<tool_response>").size(),
                    "<tool_response>") == 0 &&
                content.size() >= std::string("</tool_response>").size() &&
                content.compare(
                    content.size() - std::string("</tool_response>").size(),
                    std::string("</tool_response>").size(),
                    "</tool_response>") == 0;

            if (!is_tool_response) {
                multi_step_tool = false;
                last_query_index = index;
            }
        }
    }

    // ---------------------------------------------------------------------
    // Render messages.
    // ---------------------------------------------------------------------

    for (size_t i = 0; i < messages.size(); ++i) {
        const Message& message = messages[i];

        const bool is_first = (i == 0);
        const bool is_last = (i + 1 == messages.size());

        std::string content = message.content;

        // -----------------------------------------------------------------
        // user / system
        //
        // {% if (message.role == "user") or
        //       (message.role == "system" and not loop.first) %}
        // -----------------------------------------------------------------

        if (message.role == "user" ||
            (message.role == "system" && !is_first)) {

            out += "<|im_start|>";
            out += message.role;
            out += "\n";
            out += content;
            out += "<|im_end|>\n";

            continue;
        }

        // -----------------------------------------------------------------
        // assistant
        // -----------------------------------------------------------------

        if (message.role == "assistant") {

            std::string reasoning_content;

            // -------------------------------------------------------------
            // {% set reasoning_content = '' %}
            //
            // {% if message.reasoning_content is string %}
            //     {% set reasoning_content = message.reasoning_content %}
            // {% else %}
            //     {% if '</think>' in content %}
            //         ...
            //     {% endif %}
            // {% endif %}
            // -------------------------------------------------------------

            if (!message.reasoning_content.empty()) {
                reasoning_content = message.reasoning_content;
            }
            else {
                const std::string end_think = "</think>";

                size_t end_pos = content.find(end_think);

                if (end_pos != std::string::npos) {
                    std::string before =
                        content.substr(0, end_pos);

                    std::string after =
                        content.substr(
                            end_pos + end_think.size());

                    // Python:
                    //
                    // content.split('</think>')[0]
                    //     .rstrip('\n')
                    //     .split('<think>')[-1]
                    //     .lstrip('\n')

                    while (!before.empty() &&
                           before.back() == '\n') {
                        before.pop_back();
                    }

                    const std::string start_think = "<think>";

                    size_t start_pos = before.rfind(start_think);

                    if (start_pos != std::string::npos) {
                        reasoning_content =
                            before.substr(
                                start_pos + start_think.size());
                    }
                    else {
                        reasoning_content = before;
                    }

                    while (!reasoning_content.empty() &&
                           reasoning_content.front() == '\n') {
                        reasoning_content.erase(
                            reasoning_content.begin());
                    }

                    // Python:
                    //
                    // content.split('</think>')[-1].lstrip('\n')

                    while (!after.empty() &&
                           after.front() == '\n') {
                        after.erase(after.begin());
                    }

                    content = after;
                }
            }

            // -------------------------------------------------------------
            // {% if loop.index0 > ns.last_query_index %}
            // -------------------------------------------------------------

            if (i > last_query_index) {

                // ---------------------------------------------------------
                // if loop.last or
                //    (not loop.last and reasoning_content)
                // ---------------------------------------------------------

                if (is_last || !reasoning_content.empty()) {

                    out += "<|im_start|>";
                    out += message.role;
                    out += "\n";
                    out += "<think>\n";

                    // Jinja:
                    // reasoning_content.strip('\n')
                    //
                    // Only strips newline characters, not whitespace.
                    std::string reasoning = reasoning_content;

                    while (!reasoning.empty() &&
                           reasoning.front() == '\n') {
                        reasoning.erase(reasoning.begin());
                    }

                    while (!reasoning.empty() &&
                           reasoning.back() == '\n') {
                        reasoning.pop_back();
                    }

                    out += reasoning;
                    out += "\n</think>\n\n";

                    // content.lstrip('\n')
                    while (!content.empty() &&
                           content.front() == '\n') {
                        content.erase(content.begin());
                    }

                    out += content;
                }
                else {
                    out += "<|im_start|>";
                    out += message.role;
                    out += "\n";
                    out += content;
                }
            }
            else {
                out += "<|im_start|>";
                out += message.role;
                out += "\n";
                out += content;
            }

            // -------------------------------------------------------------
            // tool_calls
            // -------------------------------------------------------------

            if (!message.tool_calls.empty()) {

                for (size_t j = 0;
                     j < message.tool_calls.size();
                     ++j) {

                    const ToolCall& call =
                        message.tool_calls[j];

                    // Jinja:
                    //
                    // if (loop.first and content)
                    //     or (not loop.first)
                    //
                    if ((j == 0 && !content.empty()) ||
                        j != 0) {
                        out += "\n";
                    }

                    out += "<tool_call>\n";
                    out += "{\"name\": \"";
                    out += call.name;
                    out += "\", \"arguments\": ";

                    // In the template arguments are emitted directly if
                    // already a string, otherwise |tojson is applied.
                    //
                    // Since our representation stores arguments as JSON,
                    // emit it directly.
                    out += call.arguments;

                    out += "}\n";
                    out += "</tool_call>";
                }
            }

            out += "<|im_end|>\n";

            continue;
        }

        // -----------------------------------------------------------------
        // tool
        //
        // Consecutive tool messages are wrapped in one user block.
        // -----------------------------------------------------------------

        if (message.role == "tool") {

            // Jinja:
            //
            // if loop.first or
            //    messages[loop.index0 - 1].role != "tool"
            //
            if (is_first ||
                messages[i - 1].role != "tool") {

                out += "<|im_start|>user";
            }

            out += "\n<tool_response>\n";
            out += content;
            out += "\n</tool_response>";

            // Jinja:
            //
            // if loop.last or
            //    messages[loop.index0 + 1].role != "tool"
            //
            if (is_last ||
                messages[i + 1].role != "tool") {

                out += "<|im_end|>\n";
            }

            continue;
        }
    }

    // ---------------------------------------------------------------------
    // Generation prompt.
    //
    // {% if add_generation_prompt %}
    //     {{- '<|im_start|>assistant\n' }}
    //     {% if enable_thinking is defined and enable_thinking is false %}
    //         {{- '<think>\n\n</think>\n\n' }}
    //     {% endif %}
    // {% endif %}
    // ---------------------------------------------------------------------

    if (add_generation_prompt) {
        out += "<|im_start|>assistant\n";

        if (!enable_thinking) {
            out += "<think>\n\n</think>\n\n";
        }
    }

    return out;
}
