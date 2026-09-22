#pragma once

#include "ggml/Context.hpp"
#include "ggml/Scope.hpp"
#include "ggml/Runtime.hpp"
#include <set>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

// Computation is the build-phase value: a Tensor (a ggml tensor handle in
// one of the computation's contexts) plus a shared Description of the whole
// computation. The build allocates nothing and executes nothing; the result
// is a lazy description that the ExecutionRuntime executes in one pass.

struct ComputationScope {
    struct Repeat {
        size_t count; // number of iteration steps
        size_t iter; // the loop clock: the per-iteration providers read it

        // Flattened to pairs of tensors so loops and saves work seamlessly with std::vector<Tensor>
        std::vector<std::pair<Tensor, Tensor>> feedback;  
    };

    std::optional<Context> context; // the graph's scratch (scheduler-allocated, disposable)
    std::vector<Tensor> outputs;    // the graph's outputs (in the temporary context)
    std::optional<Repeat> repeat;     // the graph should repeat
    std::vector<std::pair<Tensor, Tensor>> saves;       
};

// Trait to extract tensors from T and reconstruct T from tensors.
// This allows the monad to process std::vector<Tensor> (or custom structs)
// identically to a single Tensor.
template <typename T>
struct ComputationValue {
    static T unwrap(const T& x) { return x; }
    static T wrap(const T& x) { return x; }
};

// Expansion for std::optional<T>
template <typename T>
struct ComputationValue<std::optional<T>> {
    static std::vector<Tensor> unwrap(const std::optional<T>& x) {
        if (x) return ComputationValue<T>::unwrap(*x);
        return {};
    }

    static std::optional<T> wrap(const std::vector<Tensor>& ts) {
        if (ts.empty()) return std::nullopt;
        return ComputationValue<T>::wrap(ts);
    }

    static std::optional<T> empty_like(const std::optional<T>& x) {
        if (x) return ComputationValue<T>::empty_like(*x);
        return std::nullopt;
    }
};

// Expansion for std::vector<T>
template <typename T>
struct ComputationValue<std::vector<T>> {
    static std::vector<Tensor> unwrap(const std::vector<T>& x) {
        std::vector<Tensor> res;
        for (const auto& elem : x) {
            auto sub = ComputationValue<T>::unwrap(elem);
            res.insert(res.end(), sub.begin(), sub.end());
        }
        return res;
    }

    static std::vector<T> wrap(const std::vector<Tensor>& ts) {
        std::vector<T> res;
        res.reserve(ts.size());
        // Reconstructs the vector by wrapping each individual tensor.
        // This perfectly supports std::vector<Tensor> and any T that maps to a single tensor.
        for (const auto& t : ts) {
            res.push_back(ComputationValue<T>::wrap({t}));
        }
        return res;
    }

    static std::vector<T> empty_like(const std::vector<T>& x) {
        std::vector<T> res;
        res.reserve(x.size());
        for (const auto& elem : x) {
            res.push_back(ComputationValue<T>::empty_like(elem));
        }
        return res;
    }
};

template <>
struct ComputationValue<Tensor> {
    static std::vector<Tensor> unwrap(const Tensor& t) {
        return {t};
    }

    static Tensor wrap(const std::vector<Tensor>& ts) {
        return ts.at(0);
    }

    static Tensor empty_like(const Tensor& t) {
        return Tensor::empty(t.shape(), t.dtype());
    }

    static Tensor& output(Scope scope, Tensor& t) {
        // The meta backend skips a node whose view_src is a static
        // tensor (a GGML_OP_NONE in a host buffer) and asserts that the
        // graph's last node is not such a skip (ggml_backend_meta
        // _graph_compute: i_start == n_nodes). ggml flattens view chains
        // in ggml_set_view_op, so view_src is always the base tensor. A graph
        // that ends in a view of a static tensor has no computable last node,
        // so materialize it with a copy: the dup is a compute node the meta 
        // can place, and it keeps the output in the same (replicated) distribution
        // the plan committed.
        if ((*t)->view_src != nullptr && (*t)->view_src->op == GGML_OP_NONE)
            t = t.clone();

        // Materialize output tensor by making it contiguous if needed.
        else if (!t.is_contiguous())
            t = t.contiguous();
        
        // Mark output to active runtime.
        scope.runtime().set_output(*t);

        return t;
    }
};

struct ComputationDescription {
    std::set<Context*> pinned;        // provided from outside (pinned)
    std::optional<Context> state;     // created on demand
    std::vector<ComputationScope> scopes;

    Context& context() {
        if (!state)
            state.emplace();
        return *state;
    }

    ComputationScope& add_scope() {
        scopes.push_back(ComputationScope());
        auto& r = scopes.back();
        r.context.emplace();
        return r;
    }

    ComputationScope& scope_of(const Tensor& t) {
        for (auto& op : scopes)
            for (auto cur = ggml_get_first_tensor(**op.context); cur != nullptr; cur = ggml_get_next_tensor(**op.context, cur))
                if (cur == *t)
                    return op;

        throw std::runtime_error("scope_of(): unknown Tensor: does not exist in any computational scope");
    }
};

template <class T>
class Computation {
public:
    struct VoidRef {};
    typedef std::conditional_t<std::is_void_v<T>, VoidRef, T> Ref;

    template <class Body, bool Void = std::is_void_v<T>>
    struct ScopeResult;

    template <class Body>
    struct ScopeResult<Body, true> {
        using type = std::invoke_result_t<Body&, Scope&>;
    };

    template <class Body>
    struct ScopeResult<Body, false> {
        using type = std::invoke_result_t<Body&, Scope&, T&>;
    };

    template <class Body, bool Void = std::is_void_v<T>>
    struct FoldResult;

    template <class Body>
    struct FoldResult<Body, true> {
        using type = std::invoke_result_t<Body&, Scope&, size_t&>;
    };

    template <class Body>
    struct FoldResult<Body, false> {
        using type = std::invoke_result_t<Body&, Scope&, size_t&, T&>;
    };

    template <class... Args>
    static Computation<std::vector<T>> all(const Computation<T>& first, const Args&... rest) {
        std::vector<T> values;
        values.reserve(1 + sizeof...(rest));

        values.push_back(*first);

        (
            [&] {
                if (rest.desc() != first.desc_)
                    throw std::runtime_error(
                        "all(): computations have different descriptions");

                values.push_back(*rest);
            }(),
            ...
        );

        return {
            std::move(values),
            first.desc_
        };
    }

    explicit Computation(std::set<Context*> pinned) : Computation() {
        desc_->pinned = std::move(pinned);
    }

    Computation() : ref_(), desc_() {
        desc_ = std::make_shared<ComputationDescription>();
    }

    Computation(Ref ref, std::shared_ptr<ComputationDescription> desc) : ref_(ref), desc_(desc) {

    }

    T operator *() const {
        return ref_;
    }

    T* operator ->() {
        return &ref_;
    }

    std::shared_ptr<ComputationDescription> desc() const {
        return desc_;
    }

    // Sequences this computation with a function that produces the next 
    // value from the current value. 
    // 
    // Because this is a build-phase monad, `f` is evaluated immediately 
    // at build time. The function `f` takes the underlying value (e.g., a 
    // Tensor or std::vector<Tensor>) and must return a new value.
    template <class F>
    auto bind(F&& f) {
        auto next = std::forward<F>(f)(ref_);

        return {next, desc_};
    }

    // Creates a single-execution graph.
    // The provided `body` builds a low-level tensor chain within a fresh, 
    // scheduler-allocated temporary context. The returned Computation<U> 
    // wraps the output(s) of the body. Because the temporary context is 
    // disposable after execution, use `state()` to persist outputs across 
    // multiple scopes.
    template <class Body>
    auto scope(Body body) -> Computation<typename ScopeResult<Body>::type> {
        using U = typename ScopeResult<Body>::type;

        auto& r = desc_->add_scope();
        Scope scope(*r.context);

        U out;

        if constexpr (std::is_void_v<T>) {
            out = body(scope);
        } else {
            out = body(scope, **this);
        }

        auto values = ComputationValue<U>::unwrap(out);

        for (auto& value : values)
            value = ComputationValue<std::decay_t<decltype(value)>>::output(scope, value);

        r.outputs = values;

        return {ComputationValue<U>::wrap(values), desc_};
    }

    // The Computation loop.
    // The provided `body` builds a single graph that is re-executed `count` times.
    // The body receives the scope, the loop's iteration clock (`iter`), and the current
    // computational value. Per-iteration inputs are dynamically rewritten before each
    // execution, and the body's output become an implicit state that is fed back into
    // the state cells after each step to persist the loop state.
    template <class Body>
    auto fold(size_t count, Body body) -> Computation<T> {
        using U = typename FoldResult<Body>::type;

        auto& r = desc_->add_scope();
        r.repeat = {count, 0, {}};
        Scope scope(*r.context);

        U next;

        if constexpr (std::is_void_v<T>) {
            next = body(scope, r.repeat->iter);
        } else {
            next = body(scope, r.repeat->iter, **this);
        }
        
        auto next_values = ComputationValue<U>::unwrap(next);
        auto curr_values = ComputationValue<T>::unwrap(**this);

        for (auto& value : next_values)
            value = ComputationValue<std::decay_t<decltype(value)>>::output(scope, value);
        
        r.outputs = next_values;
        
        for (auto i = 0; i < next_values.size(); ++i)
            r.repeat->feedback.push_back({next_values[i], curr_values[i]});
        
        return *this;
    }

    // Promotes a value across a scope boundary.
    // Since the scheduler's allocation is per-graph, a value produced in one 
    // scope's temporary context cannot be directly consumed by a later one.
    // This method allocates a persistent cell in the state context and 
    // registers a copy operation to transfer the value immediately after 
    // the producing graph executes.
    template <typename U = T, std::enable_if_t<!std::is_void_v<U>, int> = 0>
    Computation<U> state() {
        auto values = ComputationValue<T>::unwrap(ref_);

        // An empty value (std::nullopt or an empty vector) has no tensors
        // to persist: it maps back to itself without a state cell.
        if (values.empty())
            return {ComputationValue<T>::wrap(values), desc_};

        Scope scope(desc_->context());

        // One state cell per value tensor, in the state context. The cells
        // are the destination of the producing scope's save copies.
        std::vector<Tensor> cells;
        cells.reserve(values.size());

        for (auto& value : values)
            cells.push_back(ComputationValue<Tensor>::empty_like(value));

        for (auto i = 0; i < values.size(); ++i)
            desc_->scope_of(values[i]).saves.push_back({values[i], cells[i]});

        return {ComputationValue<T>::wrap(cells), desc_};
    }

    // Creates an initial state cell.
    // Allocates a persistent tensor in the state context that exists before 
    // any graph runs. The value is populated by the provided `provider` callback.
    template <typename F>
    Computation<Tensor> state(const Tensor::Shape& shape, F&& provider) {
        // U is deduced from the provider's return type (std::vector<U>).
        using U = typename std::invoke_result_t<F, std::mt19937&>::value_type;
        auto cell = desc_->context().create<U>(shape, Context::Provider<U>(provider));
        return {cell, desc_};
    }

private:
    Ref ref_;
    std::shared_ptr<ComputationDescription> desc_;
};
