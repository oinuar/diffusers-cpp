#pragma once

#include <stdexcept>
#include <string>
#include <nlohmann/json.hpp>
#include "./ArgumentParser.hpp"

template <>
struct ArgumentParser::parser<nlohmann::json> {
    nlohmann::json operator ()(const std::string& option, const std::string& value) const {
        nlohmann::json j;

        try {
            j = j.parse(value);
        } catch (const nlohmann::json::parse_error& error) {
            throw std::runtime_error("invalid JSON in argument " + option + ": " + error.what());
        } catch (const std::runtime_error& error) {
            throw std::runtime_error("invalid argument " + option + ": " + error.what());
        }

        return std::move(j);
    }
};
