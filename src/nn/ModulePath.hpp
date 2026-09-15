#pragma once

#include <vector>
#include <string>
#include <numeric>

struct ModulePath {
    ModulePath(const std::string& separator = ".", const std::string& seed = "")
        : seperator_(separator), seed_(seed)
    {}

    std::string operator()(const std::vector<std::string>& path, const std::string& prefix = "", const std::vector<std::string>& ext = {}) {
        auto seed = seed_;

        if (!prefix.empty()) {
            seed += seperator_;
            seed += prefix;
        }

        if (ext.empty())
            return join(path, seed);

        auto ext_path = path;

        ext_path.insert(std::end(ext_path), std::begin(ext), std::end(ext));

        return join(ext_path, seed);
    }

private:
    std::string seperator_;
    std::string seed_;

    std::string join(const std::vector<std::string>& path, const std::string& seed) {
        return std::accumulate(std::begin(path), std::end(path), seed,
            [this](const std::string& acc, const std::string& x) {
                if (acc.empty())
                    return x;

                return acc + seperator_ + x;
            }
        );
    }
};
