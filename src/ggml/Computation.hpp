#pragma once

#include "ggml/Context.hpp"
#include "ggml/Scope.hpp"
#include "ggml/Runtime.hpp"
#include <memory>
#include <optional>

// Computation is the build-phase value: a Tensor (a ggml tensor handle in
// one of the computation's contexts) plus a shared Description of the whole
// computation. The build allocates nothing and executes nothing; the result
// is a lazy description that the ExecutionRuntime executes in one pass.

struct ComputationScope {
    std::string name;
    std::optional<Context> context; // the graph's scratch (scheduler-allocated, disposable)
    std::vector<Tensor> outputs;    // the graph's outputs (in the temporary context)
    bool loop = false;
    std::size_t count = 0;
    std::size_t iter = 0;           // the loop clock: the per-iteration providers read it
    
    // Flattened to pairs of tensors so loops and saves work seamlessly with std::vector<Tensor>
    std::vector<std::pair<Tensor, Tensor>> feedback;  
    std::vector<std::pair<Tensor, Tensor>> saves;       
};

// Trait to extract tensors from T and reconstruct T from tensors.
// This allows the monad to process std::vector<Tensor> (or custom structs)
// identically to a single Tensor.
template <typename T>
struct ComputationValue;

template <>
struct ComputationValue<Tensor> {
    static std::vector<Tensor> extract(const Tensor& t) { return {t}; }
    static Tensor reconstruct(const std::vector<Tensor>& ts) { return ts.at(0); }
};

template <>
struct ComputationValue<std::vector<Tensor>> {
    static std::vector<Tensor> extract(const std::vector<Tensor>& ts) { return ts; }
    static std::vector<Tensor> reconstruct(const std::vector<Tensor>& ts) { return ts; }
};

struct ComputationDescription {
    Context* weights = nullptr;          // provided from outside (pinned)
    std::optional<Context> state;        // created on demand — by state() only
    std::vector<ComputationScope> scopes;

    Context& state_ctx() {
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
            for (ggml_tensor* cur = ggml_get_first_tensor(**op.context); cur != nullptr;
                cur = ggml_get_next_tensor(**op.context, cur))
                if (cur == *t)
                    return op;
        throw std::runtime_error("scope_of: tensor not in any computational scope");
    }
};

template <class T>
class Computation {
public:
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

    explicit Computation(Context& weights_context) : Computation() {
        desc_->weights = &weights_context;
    }

    Computation() : ref_(), desc_() {
        desc_ = std::make_shared<ComputationDescription>();
    }

    Computation(T ref, std::shared_ptr<ComputationDescription> desc) : ref_(ref), desc_(desc) {

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
    // computation from the current result. 
    // 
    // Because this is a build-phase monad, `f` is evaluated immediately 
    // at build time. The function `f` takes the underlying value (e.g., a 
    // Tensor or std::vector<Tensor>) and must return a new Computation.
    // To ensure the new computation is part of the same execution plan, 
    // `f` should capture the parent Computation (or its description) and 
    // use it to create subsequent scopes.
    template <class F>
    auto bind(F&& f) const {
        if (ComputationValue<T>::extract(ref_).empty())
            throw std::runtime_error("bind(): Computation is empty");
        
        auto next = std::forward<F>(f)(ref_);
        return next;
    }

    // Creates a single-execution graph.
    // The provided `body` builds a low-level tensor chain within a fresh, 
    // scheduler-allocated temporary context. The returned Computation<U> 
    // wraps the output(s) of the body. Because the temporary context is 
    // disposable after execution, use `state()` to persist outputs across 
    // multiple scopes.
    template<class Body>
    auto scope(Body body) -> Computation<std::invoke_result_t<Body, Scope&>> {
        using U = std::invoke_result_t<Body, Scope&>;
        auto& r = desc_->add_scope();
        Scope scope(*r.context);  
        auto out = body(scope);

        auto tensors = ComputationValue<U>::extract(out);

        for (auto& tensor : tensors) {
            // The meta backend skips a node whose view_src is a static
            // tensor (a GGML_OP_NONE in a host buffer) and asserts that the
            // graph's last node is not such a skip (ggml_backend_meta
            // _graph_compute: i_start == n_nodes). ggml flattens view chains
            // in ggml_set_view_op, so view_src is always the base tensor. A graph
            // that ends in a view of a static tensor has no computable last node,
            // so materialize it with a copy: the dup is a compute node the meta 
            // can place, and it keeps the output in the same (replicated) distribution
            // the plan committed.
            if ((*tensor)->view_src != nullptr && (*tensor)->view_src->op == GGML_OP_NONE)
                tensor = tensor.clone();

            // Materialize output tensor by making it contiguous if needed.
            else if (!tensor.is_contiguous())
                tensor = tensor.contiguous();
            
            // Mark output to active runtime.
            scope.runtime().set_output(*tensor);
        }

        r.outputs = tensors;
        return {ComputationValue<U>::reconstruct(tensors), desc_};
    }

    // The Computation loop.
    // The provided `body` builds a single graph that is re-executed `count` times.
    // The body receives the scope, the current Computation (for state access), and 
    // the loop's iteration clock (`iter`). Per-iteration inputs are dynamically 
    // rewritten before each execution, and the body's output is fed back into 
    // the state cells after each step to persist the loop state.
    template<class Body>
    Computation<T> repeat(std::size_t count, Body body) {
        auto& r = desc_->add_scope();
        r.loop = true;
        r.count = count;
        Scope scope(*r.context);  
        auto next = body(scope, **this, r.iter);
        
        auto next_tensors = ComputationValue<T>::extract(next);
        auto curr_tensors = ComputationValue<T>::extract(**this);
        
        r.outputs = next_tensors;
        
        for (std::size_t i = 0; i < next_tensors.size(); ++i)
            r.feedback.push_back({next_tensors[i], curr_tensors[i]});
        
        return *this;
    }

    // Lifts a value across a scope boundary.
    // Since the scheduler's allocation is per-graph, a value produced in one 
    // scope's temporary context cannot be directly consumed by a later one.
    // This method allocates a persistent cell in the state context and 
    // registers a copy operation to transfer the value immediately after 
    // the producing graph executes.
    Computation<T> state() {
        auto tensors = ComputationValue<T>::extract(ref_);

        if (tensors.empty())
            throw std::runtime_error("state(): Computation is empty");

        Scope scope(desc_->state_ctx());

        std::vector<Tensor> cells;
        cells.reserve(tensors.size());

        for (auto& t : tensors) {
            auto cell = Tensor::empty(t.shape(), t.dtype());
            cell.name(t.name());
            cells.push_back(cell);
        }

        for (auto i = 0; i < tensors.size(); ++i)
            desc_->scope_of(tensors[i]).saves.push_back({tensors[i], cells[i]});
        
        return {ComputationValue<T>::reconstruct(cells), desc_};
    }

    // Creates an initial state cell.
    // Allocates a persistent tensor in the state context that exists before 
    // any graph runs. The value is populated by the provided `provider` callback, 
    // abstracting away the context allocation details from the caller.
    template<class U>
    Computation<Tensor> state(const Tensor::Shape& shape, const Context::Provider<U>& provider) {
        auto cell = desc_->state_ctx().create<U>(shape, provider);
        return {cell, desc_};
    }

private:
    T ref_;
    std::shared_ptr<ComputationDescription> desc_;
};
