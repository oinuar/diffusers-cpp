#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <charconv>
#include <iostream>
#include "Tensor.hpp"

class ArgumentParser {
public:
    ArgumentParser(int argc, char** argv);

    template <typename T>
    T get(std::string_view option) const;

    const std::string& command() const {
        return command_;
    }

private:
    std::string command_;
    std::unordered_map<std::string, std::string> options_;

    template<typename T>
    static T parse_number(const std::string& value);
};

ArgumentParser::ArgumentParser(int argc, char** argv)
{
    if (argc < 2)
        throw std::runtime_error("Missing command.");

    command_ = argv[1];

    for (int i = 2; i < argc; ++i) {
        std::string key = argv[i];

        if (!key.rfind("--", 0) == 0)
            throw std::runtime_error("Expected option beginning with '--': " + key);

        // Boolean flag
        if (i + 1 == argc || std::string_view(argv[i + 1]).rfind("--", 0) == 0) {
            options_[std::move(key)] = "";
        } else {
            options_[std::move(key)] = argv[++i];
        }
    }
}

template <>
std::string ArgumentParser::get<std::string>(std::string_view option) const
{
    auto it = options_.find(std::string(option));

    if (it == options_.end())
        throw std::runtime_error("Missing required option: " + std::string(option));

    return it->second;
}

template <typename T>
T ArgumentParser::get(std::string_view option) const
{
    return parse_number<T>(get<std::string>(option));
}

template<typename T>
T ArgumentParser::parse_number(const std::string& value)
{
    T result{};

    auto [ptr, ec] = std::from_chars(
        value.data(),
        value.data() + value.size(),
        result);

    if (ec != std::errc{} || ptr != value.data() + value.size())
        throw std::runtime_error("Invalid number: " + value);

    return result;
}
