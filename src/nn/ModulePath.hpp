#pragma once

#include <vector>
#include <string>
#include <numeric>

struct ModulePath {
    ModulePath(const std::string& separator = ".", const std::string& seed = "")
        : seperator_(separator), seed_(seed)
    {}

    std::string operator()(const std::vector<std::string>& path, const std::string& prefix = "") {
        auto seed = seed_;

        if (!prefix.empty()) {
            seed += seperator_;
            seed += prefix;
        }

        return std::accumulate(std::begin(path), std::end(path), seed,
            [this](const std::string& acc, const std::string& x) {
                if (acc.empty())
                    return x;
                return acc + seperator_ + x;
            }
        );
    }

private:
    std::string seperator_;
    std::string seed_;
};
