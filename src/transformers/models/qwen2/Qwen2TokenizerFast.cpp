#include "Qwen2TokenizerFast.hpp"
#include <fstream>
#include <nlohmann/json.hpp>

using namespace tokenizers;

Qwen2TokenizerFast::Qwen2TokenizerFast(const std::filesystem::path& tokenizer_file) {
    std::ifstream file(tokenizer_file, std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error(
            "Failed to open tokenizer file: " +
            tokenizer_file.string());
    }

    std::string json(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());

    if (json.empty()) {
        throw std::runtime_error(
            "Tokenizer file is empty: " +
            tokenizer_file.string());
    }

    tokenizer_ = Tokenizer::FromBlobJSON(json);

    if (!tokenizer_) {
        throw std::runtime_error(
            "Failed to create tokenizer from: " +
            tokenizer_file.string());
    }

    file.clear();
    file.seekg(0, std::ios::beg);

    nlohmann::json j;
    file >> j;

    if (!j.contains("added_tokens")) {
        throw std::runtime_error(
            "Tokenizer JSON does not contain added_tokens");
    }

    for (const auto& token : j["added_tokens"]) {
        const std::string content = token.at("content").get<std::string>();
        const int id = token.at("id").get<int>();

        if (content == "<|im_end|>") {
            eos_token_id_ = id;
        }

        if (content == "<|endoftext|>") {
            unk_token_id_ = id;
            pad_token_id_ = id;
        }
    }

    if (eos_token_id_ < 0) {
        throw std::runtime_error(
            "Tokenizer does not define <|im_end|>");
    }

    if (unk_token_id_ < 0) {
        throw std::runtime_error(
            "Tokenizer does not define <|endoftext|>");
    }

    if (pad_token_id_ < 0) {
        throw std::runtime_error(
            "Tokenizer does not define pad token");
    }
}

std::vector<int> Qwen2TokenizerFast::encode(
    const std::string& text,
    int max_length,
    std::vector<int>* mask,
    size_t* num_real_tokens) const
{
    std::vector<int> ids =
        tokenizer_->Encode(text);

    const size_t real_tokens = ids.size();

    if (max_length > 0) {
        if (ids.size() > static_cast<size_t>(max_length)) {
            ids.resize(max_length);
        }

        while (ids.size() < static_cast<size_t>(max_length)) {
            ids.push_back(pad_token_id_);
        }
    }

    if (num_real_tokens)
        *num_real_tokens = std::min(
            real_tokens,
            max_length > 0
                ? static_cast<size_t>(max_length)
                : real_tokens);

    if (mask) {
        mask->resize(ids.size());

        const size_t n =
            std::min(real_tokens, ids.size());

        for (size_t i = 0; i < ids.size(); ++i)
            (*mask)[i] = i < n ? 1 : 0;
    }

    return ids;
}

std::string Qwen2TokenizerFast::decode(const std::vector<int>& ids) const {
    return tokenizer_->Decode(ids);
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
