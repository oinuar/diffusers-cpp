#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

class Distribution {
public:
    enum class Kind {
        Variable,
        Replicated,
        Sharded,
        Partial,
    };

private:
    struct Node {
        Kind kind = Kind::Variable;
        int axis = -1;
        uint32_t variable_id = 0;

        // Symbolic variable alias.
        std::shared_ptr<Node> parent;
    };

    std::shared_ptr<Node> node_;

    explicit Distribution(std::shared_ptr<Node> node)
        : node_(std::move(node))
    {
    }

    static std::shared_ptr<Node> make_variable()
    {
        static uint32_t next_id = 0;

        auto node = std::make_shared<Node>();
        node->kind = Kind::Variable;
        node->variable_id = next_id++;

        return node;
    }

    static std::shared_ptr<Node> make_concrete(
        Kind kind,
        int axis = -1)
    {
        auto node = std::make_shared<Node>();
        node->kind = kind;
        node->axis = axis;

        return node;
    }

    static std::shared_ptr<Node> find(
        std::shared_ptr<Node> node)
    {
        if (!node->parent)
            return node;

        node->parent = find(node->parent);
        return node->parent;
    }

    std::shared_ptr<Node> root() const
    {
        return find(node_);
    }

    static bool same_concrete(
        const Node& a,
        const Node& b)
    {
        if (a.kind != b.kind)
            return false;

        switch (a.kind) {
        case Kind::Replicated:
            return true;

        case Kind::Sharded:
        case Kind::Partial:
            return a.axis == b.axis;

        case Kind::Variable:
            return true;
        }

        return false;
    }

    static std::string to_string(const Node& node)
    {
        switch (node.kind) {
        case Kind::Variable:
            return "Variable(" +
                   std::to_string(node.variable_id) +
                   ")";

        case Kind::Replicated:
            return "Replicated";

        case Kind::Sharded:
            return "Sharded(" +
                   std::to_string(node.axis) +
                   ")";

        case Kind::Partial:
            return "Partial(" +
                   std::to_string(node.axis) +
                   ")";
        }

        return "?";
    }

    /*
     * Bind a variable to another node.
     *
     * Both nodes must already be roots.
     */
    static void bind(
        const std::shared_ptr<Node>& variable,
        const std::shared_ptr<Node>& value)
    {
        variable->parent = value;
    }

public:
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    static Distribution variable()
    {
        return Distribution(make_variable());
    }

    static Distribution replicated()
    {
        return Distribution(
            make_concrete(Kind::Replicated));
    }

    static Distribution sharded(int axis)
    {
        if (axis < 0)
            throw std::invalid_argument(
                "Sharded axis must be >= 0");

        return Distribution(
            make_concrete(Kind::Sharded, axis));
    }

    static Distribution partial(int axis)
    {
        if (axis < 0)
            throw std::invalid_argument(
                "Partial axis must be >= 0");

        return Distribution(
            make_concrete(Kind::Partial, axis));
    }

    // ------------------------------------------------------------------
    // Properties
    // ------------------------------------------------------------------

    Kind kind() const
    {
        return root()->kind;
    }

    int axis() const
    {
        return root()->axis;
    }

    bool is_variable() const
    {
        return kind() == Kind::Variable;
    }

    bool is_replicated() const
    {
        return kind() == Kind::Replicated;
    }

    bool is_sharded() const
    {
        return kind() == Kind::Sharded;
    }

    bool is_partial() const
    {
        return kind() == Kind::Partial;
    }

    bool is_resolved() const
    {
        return !is_variable();
    }

    Distribution resolved() const
    {
        return Distribution(root());
    }

    void require_resolved() const
    {
        if (is_variable()) {
            throw std::runtime_error(
                "Distribution has not been resolved: " +
                str());
        }
    }

    std::string str() const
    {
        return to_string(*root());
    }

    // ------------------------------------------------------------------
    // Strict unification
    // ------------------------------------------------------------------

    /*
     * Strict unification.
     *
     * The two distributions must describe exactly the same layout.
     *
     * Variable + anything
     *     -> variable is bound to anything
     *
     * Sharded(0) + Sharded(0)
     *     -> OK
     *
     * Sharded(0) + Sharded(1)
     *     -> conflict
     *
     * Replicated + Sharded(0)
     *     -> conflict
     */
    void unify(const Distribution& other)
    {
        auto a = root();
        auto b = other.root();

        if (a == b)
            return;

        // Variable + Variable.
        if (a->kind == Kind::Variable &&
            b->kind == Kind::Variable) {

            bind(a, b);
            return;
        }

        // Variable + Concrete.
        if (a->kind == Kind::Variable) {
            bind(a, b);
            return;
        }

        // Concrete + Variable.
        if (b->kind == Kind::Variable) {
            bind(b, a);
            return;
        }

        // Concrete + Concrete.
        if (!same_concrete(*a, *b)) {
            throw std::runtime_error(
                "Distribution conflict: " +
                to_string(*a) +
                " vs " +
                to_string(*b));
        }
    }

    // ------------------------------------------------------------------
    // Elementwise unification
    // ------------------------------------------------------------------

    /*
     * Unification rule for elementwise operations.
     *
     * Elementwise operations operate independently on every local
     * element, so a replicated operand can participate in any
     * distributed operation.
     *
     * Examples:
     *
     *   Replicated + Replicated
     *       -> Replicated
     *
     *   Sharded(0) + Replicated
     *       -> Sharded(0)
     *
     *   Replicated + Sharded(1)
     *       -> Sharded(1)
     *
     *   Sharded(0) + Sharded(0)
     *       -> Sharded(0)
     *
     *   Sharded(0) + Sharded(1)
     *       -> conflict
     *
     *   Partial(0) + Partial(0)
     *       -> Partial(0)
     */
    static Distribution elementwise(
        Distribution a,
        Distribution b)
    {
        // --------------------------------------------------------------
        // Replicated is the identity distribution for elementwise ops.
        // --------------------------------------------------------------

        if (a.is_replicated())
            return b;

        if (b.is_replicated())
            return a;

        // --------------------------------------------------------------
        // Both unresolved.
        //
        // There is no information yet telling us which distribution
        // the result should have.
        //
        // Tie them together so they represent the same distribution.
        // --------------------------------------------------------------

        if (a.is_variable() &&
            b.is_variable()) {

            a.unify(b);
            return a;
        }

        // --------------------------------------------------------------
        // One unresolved.
        //
        // The concrete side determines the result.
        // --------------------------------------------------------------

        if (a.is_variable()) {
            a.unify(b);
            return a;
        }

        if (b.is_variable()) {
            b.unify(a);
            return b;
        }

        // --------------------------------------------------------------
        // Both distributed.
        //
        // They must describe the same partition.
        // --------------------------------------------------------------

        a.unify(b);
        return a;
    }

    /*
     * Convenience member form.
     *
     *     auto result = a.unify_elementwise(b);
     */
    Distribution unify_elementwise(
        const Distribution& other) const
    {
        return elementwise(*this, other);
    }

    // ------------------------------------------------------------------
    // Comparison
    // ------------------------------------------------------------------

    bool operator==(const Distribution& other) const
    {
        auto a = root();
        auto b = other.root();

        if (a == b)
            return true;

        if (a->kind == Kind::Variable ||
            b->kind == Kind::Variable)
            return false;

        return same_concrete(*a, *b);
    }

    bool operator!=(const Distribution& other) const
    {
        return !(*this == other);
    }
};
