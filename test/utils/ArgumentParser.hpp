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

    template<typename T, typename Parser>
    class iterator {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = const T*;
        using reference = const T&;

        iterator() : it_(), parser_() {}

        explicit iterator(
            std::unordered_multimap<std::string, std::string>::const_iterator it,
            Parser parser)
            : it_(it), parser_(std::move(parser))
        {}

        T operator*() const {
            return parser_(it_->first, it_->second);
        }

        iterator& operator++() {
            ++it_;
            return *this;
        }

        iterator operator++(int) {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }

        bool operator==(const iterator<T, Parser>& other) const { return it_ == other.it_; }
        bool operator!=(const iterator<T, Parser>& other) const { return it_ != other.it_; }

    private:
        std::unordered_multimap<std::string, std::string>::const_iterator it_;
        Parser parser_;
    };

    ArgumentParser(int argc, char** argv);

    template <typename T, class Parser = parser<T>>
    T get_one(std::string_view option, Parser parser = Parser()) const;

    template <typename T, class Parser = parser<T>>
    std::optional<T> get_optional(std::string_view option, Parser parser = Parser()) const;

    template <typename T, class Parser = parser<T>>
    std::vector<T> get_many(std::string_view option, Parser parser = Parser()) const;

    template <typename T, class Parser = parser<T>>
    std::pair<iterator<T, Parser>, iterator<T, Parser>> get(std::string_view option, Parser parser = Parser()) const;

    const std::string& command() const {
        return command_;
    }

private:
    std::string command_;
    std::unordered_multimap<std::string, std::string> options_;
};

ArgumentParser::ArgumentParser(int argc, char** argv) {
    if (argc < 2)
        throw std::runtime_error("Missing command.");

    command_ = argv[1];

    for (int i = 2; i < argc; ++i) {
        std::string key = argv[i];

        if (!key.rfind("--", 0) == 0)
            throw std::runtime_error("Expected option beginning with '--': " + key);

        // Boolean flag
        if (i + 1 == argc || std::string_view(argv[i + 1]).rfind("--", 0) == 0) {
            options_.insert(std::make_pair(std::move(key), ""));
        } else {
            options_.insert(std::make_pair(std::move(key), argv[++i]));
        }
    }
}

template <typename T, class Parser>
T ArgumentParser::get_one(std::string_view option, Parser parser) const {
    auto [start, end] = get<T>(option, parser);

    if (start == end)
        throw std::runtime_error("Missing option: " + std::string(option));

    auto value = *(start++);

    if (start != end)
        throw std::runtime_error("Multiple options: " + std::string(option));

    return std::move(value);
}

template <typename T, class Parser>
std::optional<T> ArgumentParser::get_optional(std::string_view option, Parser parser) const {
    auto [start, end] = get<T>(option, parser);

    if (start == end)
        return std::nullopt;

    auto value = *(start++);

    if (start != end)
        throw std::runtime_error("Multiple options: " + std::string(option));

    return value;
}

template <typename T, class Parser>
std::vector<T> ArgumentParser::get_many(std::string_view option, Parser parser) const {
    auto [start, end] = get<T>(option, parser);
    std::vector<T> result;

    for (auto it = start; it != end; ++it)
        result.push_back(*it);

    return std::move(result);
}

template <typename T, class Parser>
std::pair<ArgumentParser::iterator<T, Parser>, ArgumentParser::iterator<T, Parser>> ArgumentParser::get(
    std::string_view option, Parser parser) const
{
    auto [start, end] = options_.equal_range(std::string(option));

    return std::make_pair(iterator<T, Parser>(start, std::move(parser)), iterator<T, Parser>(end, std::move(parser)));
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
