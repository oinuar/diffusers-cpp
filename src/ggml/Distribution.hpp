#pragma once

#include <cstdint>
#include <functional>
#include <string>

enum class DistributionKind {
    Replicated,
    Sharded,
    Partial,
};

class Distribution {
public:
    Distribution()
        : kind_(DistributionKind::Replicated), axis_(-1) {}

    static Distribution replicated() {
        return Distribution(DistributionKind::Replicated, -1);
    }

    static Distribution sharded(int axis) {
        return Distribution(DistributionKind::Sharded, axis);
    }

    static Distribution partial(int axis) {
        return Distribution(DistributionKind::Partial, axis);
    }

    DistributionKind kind() const {
        return kind_;
    }

    int axis() const {
        return axis_;
    }

    bool is_replicated() const {
        return kind_ == DistributionKind::Replicated;
    }

    bool is_sharded() const {
        return kind_ == DistributionKind::Sharded;
    }

    bool is_partial() const {
        return kind_ == DistributionKind::Partial;
    }

    bool operator==(const Distribution& other) const {
        return kind_ == other.kind_ && axis_ == other.axis_;
    }

    bool operator!=(const Distribution& other) const {
        return !(*this == other);
    }

    std::string to_string() const {
        switch (kind_) {
        case DistributionKind::Replicated:
            return "R";

        case DistributionKind::Sharded:
            return "S(" + std::to_string(axis_) + ")";

        case DistributionKind::Partial:
            return "P(" + std::to_string(axis_) + ")";
        }

        return "?";
    }

private:
    Distribution(DistributionKind kind, int axis)
        : kind_(kind), axis_(axis) {}

    DistributionKind kind_;
    int axis_;
};

struct DistributionHash {
    size_t operator()(const Distribution& d) const {
        return std::hash<int>()(static_cast<int>(d.kind())) ^
               (std::hash<int>()(d.axis()) << 1);
    }
};
