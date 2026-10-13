#pragma once

#include <ggml.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Tensor quantization spec with the same grammar as the --type option of
// utils/convert-model/src/convert_hf_safetensors_to_gguf.py:
//
//   "Q8_0"                            a single type name for every tensor
//   '{"*": "Q8_0", "foo": "BF16"}'    per-tensor object: tensor name
//                                     pattern -> type name, applied to
//                                     every component
//   '{"transformer": "Q8_0",          per-component object: component
//    "vae": {"*": "F32"}, ...}'       name -> type name or per-tensor
//                                     object; the "*" entry is the default
//                                     for components without an explicit
//                                     entry
//   "@map.json"                       one of the above JSON objects read
//                                     from a file
//
// An empty value disables quantization: every tensor keeps its original
// dtype.
//
// Per-tensor objects: an exact tensor name takes precedence over patterns;
// patterns use fnmatch syntax (*, ?, [seq]) and when several match, the
// most specific one (the longest pattern) wins, ties in the order they
// appear. Tensors matching nothing fall back to the component's default
// type, and to their original dtype if there is none.
class QuantSpec {
public:
    struct TypeSpec {
        std::optional<ggml_type> type;
        std::vector<std::pair<std::string, ggml_type>> patterns;
    };

    QuantSpec() = default;

    static QuantSpec parse(const std::string& value) {
        QuantSpec spec;

        auto text = trim(value);

        if (text.empty())
            return spec;

        std::string json_text;

        if (text.front() == '@') {
            auto path = text.substr(1);

            std::error_code ec;
            if (!std::filesystem::is_regular_file(path, ec) || ec)
                throw std::runtime_error("--quant file not found: " + path);

            std::ifstream file(path);

            if (!file)
                throw std::runtime_error("Cannot read --quant file: " + path);

            json_text.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        }
        else if (text.front() == '{') {
            json_text = text;
        }
        else {
            // A bare quantization type name applied to every tensor
            spec.type_spec = TypeSpec{ from_name(text), {} };
            return spec;
        }

        nlohmann::ordered_json obj;

        try {
            obj = nlohmann::ordered_json::parse(json_text);
        }
        catch (const nlohmann::json::parse_error& error) {
            throw std::runtime_error("--quant is not valid JSON: " + std::string(error.what()));
        }

        if (!obj.is_object())
            throw std::runtime_error("--quant JSON must be an object of type names or pattern objects");

        bool per_component = false;

        for (auto it = obj.begin(); it != obj.end(); ++it) {
            if (it.value().is_object())
                per_component = true;
            else if (!it.value().is_string())
                throw std::runtime_error("--quant value for key '" + it.key() + "' must be a type name or a pattern object");
        }

        if (per_component) {
            for (auto it = obj.begin(); it != obj.end(); ++it) {
                auto type_spec = parse_type_map(it.value());

                if (it.key() == "*") {
                    if (spec.fallback)
                        throw std::runtime_error("--quant has multiple '*' component entries");

                    spec.fallback = std::move(type_spec);
                }
                else {
                    spec.components.emplace_back(it.key(), std::move(type_spec));
                }
            }
        }
        else {
            spec.type_spec = parse_type_map(obj);
        }

        return spec;
    }

    bool enabled() const {
        return type_spec || fallback || !components.empty();
    }

    /** @brief Resolves the quantization type of a tensor, or std::nullopt
     *  if the tensor keeps its original dtype.
     *
     *  @param name      the tensor name inside its component (dotted, without
     *                   the component prefix)
     *  @param component the component prefix ("" for a standalone model)
     */
    std::optional<ggml_type> resolve(const std::string& name, const std::string& component = "") const {
        const TypeSpec* type_spec = nullptr;

        if (components.empty()) {
            if (type_spec)
                type_spec = &*type_spec;
        }
        else {
            for (auto& [name, spec] : components) {
                if (name == component) {
                    type_spec = &spec;
                    break;
                }
            }

            if (!type_spec && fallback)
                type_spec = &*fallback;
        }

        if (!type_spec)
            return std::nullopt;

        // 1. An exact tensor name entry of the pattern object
        for (auto& [pattern, type] : type_spec->patterns) {
            if (pattern == name)
                return type;
        }

        // 2. The most specific (longest) matching pattern; ties in the
        //    order the patterns appear
        ggml_type best = GGML_TYPE_COUNT;
        int best_len = -1;

        for (auto& [pattern, type] : type_spec->patterns) {
            auto len = static_cast<int>(pattern.size());

            if (len > best_len && fnmatch(pattern, name)) {
                best = type;
                best_len = len;
            }
        }

        if (best != GGML_TYPE_COUNT)
            return best;

        // 3. The component-wide default type
        if (type_spec->type)
            return *type_spec->type;

        // 4. The tensor's original dtype
        return std::nullopt;
    }

private:
    std::optional<TypeSpec> type_spec;     // bare type name or per-tensor patterns
    std::vector<std::pair<std::string, TypeSpec>> components;
    std::optional<TypeSpec> fallback;      // the "*" component entry

    static TypeSpec parse_type_map(const nlohmann::ordered_json& obj) {
        TypeSpec spec;

        if (obj.is_string())
            spec.type = from_name(obj.get<std::string>());

        for (auto it = obj.begin(); it != obj.end(); ++it) {
            if (!it.value().is_string())
                throw std::runtime_error("--quant value for pattern '" + it.key() + "' must be a quantization type name");

            spec.patterns.emplace_back(it.key(), from_name(it.value().get<std::string>()));
        }

        return spec;
    }

    static std::string trim(const std::string& text) {
        auto begin = text.find_first_not_of(" \t\r\n");

        if (begin == std::string::npos)
            return "";

        auto end = text.find_last_not_of(" \t\r\n");

        return text.substr(begin, end - begin + 1);
    }

    // Maps a type name (e.g. "Q8_0") to its ggml_type. The names are the
    // GGMLQuantizationType names of the gguf Python package that
    // utils/convert-model uses: the ggml type names with all letters
    // uppercased ("q8_0" -> "Q8_0", "bf16" -> "BF16", ...).
    static ggml_type from_name(const std::string& name) {
        static const std::map<std::string, ggml_type> names = [] {
            std::map<std::string, ggml_type> m;

            for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
                auto ggml_name = ggml_type_name(static_cast<ggml_type>(i));

                // Skip the deprecated / removed placeholder types
                if (std::string(ggml_name).find(' ') != std::string::npos)
                    continue;

                m.emplace(upper_letters(ggml_name), static_cast<ggml_type>(i));
            }

            return m;
        }();

        auto it = names.find(upper_letters(name));


        if (it == names.end()) {
            std::string valid;

            for (auto& [n, _] : names)
                valid += (valid.empty() ? "" : ", ") + n;

            throw std::runtime_error("Unknown quantization type '" + name + "' in --quant. Valid types: " + valid);
        }

        return it->second;
    }

    // Uppercases the letters of a type name, leaving digits and
    // underscores untouched; the GGMLQuantizationType names of the gguf
    // package are the ggml type names uppercased this way.
    static std::string upper_letters(const std::string& text) {
        std::string out;

        for (auto c : text)
            out += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));

        return out;
    }
    // ------------------------------------------------------------------
    // fnmatch with Python fnmatch.fnmatchcase semantics: '*' matches
    // anything, '?' matches one character, '[seq]' matches a character in
    // seq ('!' negates the set, 'a-c' ranges are supported).
    // ------------------------------------------------------------------
    static bool fnmatch(const std::string& pattern, const std::string& text) {
        return match(pattern, 0, text, 0);
    }

    static bool match(const std::string& p, size_t pi, const std::string& t, size_t ti) {
        if (pi == p.size())
            return ti == t.size();

        if (p[pi] == '*') {
            if (match(p, pi + 1, t, ti))
                return true;

            for (size_t i = ti; i < t.size(); ++i)
                if (match(p, pi + 1, t, i + 1))
                    return true;

            return false;
        }

        if (ti == t.size())
            return false;

        if (p[pi] == '?')
            return match(p, pi + 1, t, ti + 1);

        if (p[pi] == '[') {
            size_t end = 0;

            if (parse_char_class(p, pi, end)) {
                if (char_in_class(p, pi, end, t[ti]))
                    return match(p, end, t, ti + 1);

                return false;
            }

            // No closing ']': the '[' is a literal character
            return t[ti] == '[' && match(p, pi + 1, t, ti + 1);
        }

        return t[ti] == p[pi] && match(p, pi + 1, t, ti + 1);
    }

    // Parses the character class at p[pi] (p[pi] == '[') and returns the
    // index just past the closing ']' in end. A ']' right after '[' (or
    // '[!') is a literal member of the set and does not close it. Returns
    // false if there is no closing ']' (the '[' is a literal then).
    static bool parse_char_class(const std::string& p, size_t pi, size_t& end) {
        size_t i = pi + 1;

        if (i < p.size() && p[i] == '!')
            ++i;

        if (i < p.size() && p[i] == ']')
            ++i;

        if (i >= p.size())
            return false;

        for (; i < p.size(); ++i) {
            if (p[i] == ']') {
                end = i + 1;
                return true;
            }
        }

        return false;
    }

    // True if the character c is a member of the class parsed at p[pi]
    // (the members are p[pi+1 .. end-2]; p[end-1] is the closing ']').
    static bool char_in_class(const std::string& p, size_t pi, size_t end, char c) {
        size_t i = pi + 1;
        bool negated = false;

        if (i + 1 < end && p[i] == '!') {
            negated = true;
            ++i;
        }

        unsigned value = static_cast<unsigned char>(c);
        bool in = false;

        for (; i + 1 < end; ++i) {
            unsigned lo = static_cast<unsigned char>(p[i]);

            // A range 'lo-hi'; a '-' at the end of the set is literal
            if (p[i + 1] == '-' && i + 2 < end - 1) {
                unsigned hi = static_cast<unsigned char>(p[i + 2]);

                if (value >= lo && value <= hi)
                    in = true;

                i += 2;
            }
            else if (value == lo) {
                in = true;
            }
        }

        return negated ? !in : in;
    }
};
