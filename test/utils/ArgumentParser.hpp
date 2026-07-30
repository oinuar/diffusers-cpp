#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <charconv>
#include <iostream>
#include <functional>
#include <type_traits>

class ArgumentParser {
public:
    template <typename T>
    struct parser {
        T operator ()(const std::string& option, const std::string& value) const;
    };

    typedef std::unordered_multimap<std::string, std::string>::const_iterator iterator;

    ArgumentParser(int argc, char** argv);

    template <typename T, class Parser = parser<T>>
    T get_one(std::string_view option, Parser parser = Parser()) const;

    template <typename T, class Parser = parser<T>>
    std::optional<T> get_optional(std::string_view option, Parser parser = Parser()) const;

    template <typename T, class Parser = parser<T>>
    std::vector<T> get_many(std::string_view option, Parser parser = Parser()) const;

    std::pair<iterator, iterator> get(std::string_view option) const;

    const std::string& get(const size_t& index) const;

private:
    std::string command_;
    std::unordered_multimap<std::string, std::string> options_;
    std::vector<std::string> positional_;
};

inline
ArgumentParser::ArgumentParser(int argc, char** argv) : options_(), positional_() {
    for (int i = 1; i < argc; ++i) {
        std::string_view arg{argv[i]};

        if (arg.rfind("--") == 0) {
            if (arg.size() == 2)
                throw std::runtime_error("Invalid option '--'.");

            std::string key(arg);

            // --key value
            if (i + 1 < argc && std::string_view(argv[i + 1]).rfind("--") != 0)
                options_.emplace(std::move(key), argv[++i]);

            // --flag
            else
                options_.emplace(std::move(key), "");
        }
        else
            positional_.emplace_back(arg);
    }
}

template <typename T, class Parser>
T ArgumentParser::get_one(std::string_view option, Parser parser) const {
    auto [start, end] = get(option);

    if (start == end)
        throw std::runtime_error("Missing option: " + std::string(option));

    auto it = start++;

    if (start != end)
        throw std::runtime_error("Multiple options: " + std::string(option));

    return std::move(parser(it->first, it->second));
}

template <typename T, class Parser>
std::optional<T> ArgumentParser::get_optional(std::string_view option, Parser parser) const {
    auto [start, end] = get(option);

    if (start == end)
        return std::nullopt;

    auto it = start++;

    if (start != end)
        throw std::runtime_error("Multiple options: " + std::string(option));

    return std::move(parser(it->first, it->second));
}

template <typename T, class Parser>
std::vector<T> ArgumentParser::get_many(std::string_view option, Parser parser) const {
    auto [start, end] = get(option);
    std::vector<T> result;

    // TODO: unordered_map returns these options in reverse order. Bad since order
    // is unspecified anyway and may change. Need to fix to keep proper (argument) ordering!
    for (auto it = start; it != end; ++it)
        result.emplace(result.begin(), parser(it->first, it->second));

    return std::move(result);
}

inline
std::pair<ArgumentParser::iterator, ArgumentParser::iterator> ArgumentParser::get(std::string_view option) const {
    auto [start, end] = options_.equal_range(std::string(option));

    return std::make_pair(start, end);
}

inline
const std::string& ArgumentParser::get(const size_t& index) const {
    if (index >= positional_.size())
        throw std::runtime_error("Missing argument at position " + std::to_string(index));

    return positional_.at(index);
}

template <typename T>
T ArgumentParser::parser<T>::operator()(const std::string& option, const std::string& value) const {
    T result{};
    auto [ptr, ec] = std::from_chars(
        value.data(), value.data() + value.size(), result);

    if (ec != std::errc{} || ptr != value.data() + value.size())
        throw std::runtime_error("Invalid option: " + option + ": invalid number: " + value);

    return result;
}

template <>
struct ArgumentParser::parser<bool> {
    bool operator ()(const std::string& option, const std::string& value) const {
        if (value == "true") return true;
        if (value == "false") return false;
        throw std::runtime_error("Invalid option: " + option + ": invalid boolean: " + value);
    }
};

template <>
struct ArgumentParser::parser<std::string> {
    std::string operator ()(const std::string&, const std::string& value) const {
        return value;
    }
};

#include "./TensorParser.hpp"
#include "./ShapeParser.hpp"
#include "./SliceParser.hpp"
#include "./JsonParser.hpp"
