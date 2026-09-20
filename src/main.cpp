// ============================================================================
// PoC: the computational monad on real ggml (low-level API), single CPU backend
// ============================================================================
//
// A standalone model of the reworked Flux2KleinPipeline (src/diffusers/
// pipelines/flux2/Flux2KleinPipeline.cpp) with the ggml stand-ins removed:
// the Tensor / Context / Scope wrappers are built directly on the low-level
// ggml API (ggml_init, ggml_new_tensor, ggml_add, ggml_build_forward_expand,
// ggml_backend_sched_*, ...) — not on the project's ggml:: wrappers in
// src/ggml/, which this PoC is meant to replace. Execution is real ggml on a
// single CPU backend (ggml_backend_cpu_init). The modules are reduced to
// their minimal form (no VAE / text encoder / transformer: an Adder doing
// 1 + 2 and a counter step) and the math stays trivial.
//
// The monad itself is unchanged (Computation<T>, bind, output, repeat, state):
//
//   bind    the Kleisli chain (join): the continuation receives the value
//           and returns the next computation's output
//   output  a single-execution graph: the body (Scope, ...) builds the
//           low-level ggml tensor chain in a fresh temporary context and
//           returns the graph's output
//   state   the door to the state context: state(value) copies a graph's
//           output into a state cell right after the graph executes;
//           state(desc, shape, provider) creates a state cell up front
//   repeat  the body (Scope, state, iter) is built once (one ggml graph)
//           and re-executed N times; the per-iteration inputs are re-bound
//           through the loop's iter clock, and the state is fed back into
//           the state cell after each iteration
//
// Memory model — three context classes (as before):
//
//   weights    provided from outside (from_pretrained), allocated once by
//              ggml_backend_alloc_ctx_tensors (one aligned buffer for the
//              whole context — the model of
//              ggml_backend_alloc_ctx_tensors_from_buft). Pinned, never
//              freed.
//   state      created on demand by state(): the values that cross graph
//              boundaries (the initial counter cell, the sum, the final
//              result). Allocated once in the same one-shot pass; lives for
//              the whole generation. It exists because the scheduler's
//              allocation is per-graph (not shared between ggml graphs).
//   temporary  one per graph output: created with no_alloc, allocated per
//              graph by the scheduler's galloc (ggml_backend_sched_alloc_
//              graph), disposed (ggml_reset) after the graph executes. The
//              compute buffer is persistent and reused across graphs (one
//              buffer for the whole generation).
//
// The loop: the graph is a single allocation with multiple computations
// (the ggml_backend_sched contract). Before each re-execution the
// re-bindable inputs are rewritten (ggml_backend_tensor_set — their
// providers run through the loop's iter clock); after each one the state is
// copied back into the state cell (ggml_backend_tensor_copy) — the feedback
// the old imperative denoise loop was missing.
//
// What this replaces in the current code (same as the stand-in PoC):
//
//   - the four per-stage Computation objects (each: allocate + plan +
//     execute one graph) -> one Executor: one allocation pass, one
//     execution pass over the outputs.
//   - the imperative denoise loop (the latents input never updated) -> loop:
//     the state is fed back after each step.
//   - the incremental allocator (the weights re-streamed per graph) -> the
//     one-shot non-incremental allocator (the weights streamed once).
//
// Mapping (PoC -> project):
//
//   Tensor/Shape/Context/Scope    ggml::Tensor / Shape / Context / Scope
//                                 (the migration target of these wrappers)
//   Adder/CounterStep             the nn::Module trees (Module::forward)
//   Pipeline::compute             Flux2KleinPipeline::operator()
//   load_weights                  from_pretrained + GGUFLoaderVisitor
//   the one-shot allocator        the simplified ggml::Allocator
//   the scheduler                 ggml_backend_sched (the galloc)
//   the executor                  the replacement of the Computation class
//
// Build & run:
//   cmake --build build -j8
//   uv run build/bin/diffusers-cli

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// ============================================================================
// Shape (PyTorch order; mirrors src/ggml/Tensor.hpp's Shape)
// ============================================================================

class Tensor;

// PyTorch-order shape: the first index is the slowest (outermost) dimension;
// internally ne[0] is the fastest (innermost) dimension (as in the real
// Shape).
class Shape {
public:
    Shape() = default;
    explicit Shape(int64_t rank) : rank_(rank) {}
    Shape(std::initializer_list<int64_t> list) : rank_(static_cast<int64_t>(list.size())) {
        std::size_t i = 0;
        for (auto it = std::rbegin(list); it != std::rend(list); ++it)
            ne_[i++] = *it;
    }

    int64_t operator[](int64_t i) const {
        assert(i >= 0 && i < rank_);
        return ne_[rank_ - 1 - i];
    }

    int64_t rank() const { return rank_; }
    const int64_t* data() const { return ne_.data(); }  // ggml order: ne[0] = fastest

    int64_t numel() const {
        int64_t n = 1;
        for (int64_t i = 0; i < rank_; ++i)
            n *= ne_[i];
        return n;
    }

    bool operator==(const Shape& o) const { return ne_ == o.ne_ && rank_ == o.rank_; }
    bool operator!=(const Shape& o) const { return !(*this == o); }

    std::string to_string() const {
        std::string s = "(";
        for (int64_t i = 0; i < rank_; ++i) {
            if (i)
                s += ", ";
            s += std::to_string((*this)[i]);
        }
        return s + ")";
    }

private:
    friend class Tensor;
    std::array<int64_t, 4> ne_{};  // ne[0] = the fastest (innermost) dimension
    int64_t rank_ = 0;
};

// ============================================================================
// Tensor (wraps ggml_tensor*; mirrors the minimal API of src/ggml/Tensor.hpp)
// ============================================================================

class Context;
// ============================================================================
// Scope (mirrors src/ggml/Scope.hpp): the thread-local "current context"
// ============================================================================

class Scope {
public:
    explicit Scope(Context& context) : frame_(std::make_shared<Frame>(current_)) { current_ = &context; }
    Scope(const Scope&) = default;
    Scope& operator=(const Scope&) = delete;
    Scope(Scope&&) = delete;
    Scope& operator=(Scope&&) = delete;
    ~Scope() {
        if (frame_.unique())
            current_ = frame_->previous;
    }

    static Context& context() {
        if (!current_)
            throw std::runtime_error("Scope::context(): no active Scope");
        return *current_;
    }

private:
    struct Frame {
        Context* previous;
        explicit Frame(Context* p) : previous(p) {}
    };
    std::shared_ptr<Frame> frame_;
    inline static thread_local Context* current_ = nullptr;
};

class Tensor {
public:
    template<class T>
    struct DType;

    Tensor() = default;

    // Wraps a ggml tensor, inferring the shape (as in the real Tensor).
    // The Tensor does not own the pointer — it is valid only while the
    // tensor's ggml context is alive.
    explicit Tensor(ggml_tensor* t) : t_(t), shape_() {
        if (t_ != nullptr) {
            shape_ = Shape({t_->ne[3], t_->ne[2], t_->ne[1], t_->ne[0]});
            shape_.rank_ = ggml_n_dims(t_);
        }
    }

    ggml_tensor* operator *() const { return t_; }
    bool valid() const { return t_ != nullptr; }
    operator bool() const { return t_ != nullptr; }

    const Shape& shape() const { return shape_; }
    ggml_type dtype() const { return t_->type; }
    int64_t numel() const { return ggml_nelements(t_); }
    std::size_t nbytes() const { return ggml_nbytes(t_); }
    const char* name() const { return ggml_get_name(t_); }
    const Tensor& name(const char* n) const { ggml_set_name(t_, n); return *this; }
    const Tensor& name(const std::string& n) const { return name(n.c_str()); }

    // Creates a tensor in the active scope's context (metadata only — the
    // data is written later: by the one-shot allocator for the long-lived
    // contexts, by the scheduler's galloc for the temporary ones).
    template<class T>
    static Tensor empty(const Shape& shape);

    // Marks the tensor as a graph input (ggml_set_input) — the bound inputs
    // (create / value).
    Tensor input() const;

    // Tensor ops (as in the real Tensor): each records a node in the active
    // scope's context.
    Tensor operator+(const Tensor& rhs) const;
    Tensor operator*(const Tensor& rhs) const;
    Tensor operator-(const Tensor& rhs) const;

private:
    ggml_tensor* t_ = nullptr;
    Shape shape_;
};

template<>
struct Tensor::DType<float>   { static constexpr ggml_type value = GGML_TYPE_F32; };
template<>
struct Tensor::DType<int32_t> { static constexpr ggml_type value = GGML_TYPE_I32; };

// ============================================================================
// Context (wraps ggml_context; mirrors the minimal API of src/ggml/Context.hpp)
// ============================================================================

class Context {
public:
    using Provider = std::function<std::vector<std::byte>(std::mt19937&)>;

    struct Binding {
        Provider provider;
        bool once;
        bool unbound = false;
    };

    // The ggml context is created with no_alloc: it owns only the tensor
    // metadata (and the graph's cgraph, see the capacity). The tensor data
    // comes later — one-shot for the long-lived contexts (allocate), per
    // graph by the scheduler for the temporary ones.
    explicit Context(std::string name, std::size_t capacity = GGML_DEFAULT_GRAPH_SIZE)
        : name_(std::move(name)),
          metadata_(ggml_tensor_overhead() * capacity + ggml_graph_overhead()),
          capacity_(capacity)
    {
        ctx_ = ggml_init({metadata_.size(), metadata_.data(), /*no_alloc=*/true});
        assert(ctx_ != nullptr);
    }

    Context(Context&& other)
        : name_(std::move(other.name_))
        , metadata_(std::move(other.metadata_))
        , ctx_(other.ctx_)
        , buffer_(other.buffer_)
        , capacity_(other.capacity_)
        , bindings_(std::move(other.bindings_))
    {
        other.buffer_ = nullptr;
        other.ctx_ = nullptr;
    }

    ~Context() {
        if (buffer_ != nullptr)
            ggml_backend_buffer_free(buffer_);
        if (ctx_ != nullptr)
            ggml_free(ctx_);
    }

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context& operator=(Context&&) = delete;

    const std::string& name() const { return name_; }
    ggml_context* operator *() const { return ctx_; }
    std::size_t capacity() const { return capacity_; }

    bool has_buffer() const { return buffer_ != nullptr; }
    ggml_backend_buffer_t buffer() const { return buffer_; }
    const std::map<ggml_tensor*, Binding>& bindings() const { return bindings_; }

    // A bare tensor (metadata only, no binding) — the state cells created by
    // state(): storage the executor writes into via copy.
    Tensor tensor(ggml_type type, const Shape& shape) {
        Scope scope(*this);
        return Tensor{ggml_new_tensor(ctx_, type, static_cast<int>(shape.rank()), shape.data())};
    }

    // Binding (as in the real Context::bind<T>): the provider is called at
    // execution time and the value is written into the tensor's buffer
    // (ggml_backend_tensor_set).
    template<class T>
    void bind(Tensor t, const std::function<std::vector<T>(std::mt19937&)>& provider, bool once = false) {
        bindings_[*t] = {
            [provider](std::mt19937& rng) {
                auto values = provider(rng);
                if constexpr (std::is_same_v<T, std::byte>)
                    return std::move(values);
                std::vector<std::byte> bytes(values.size() * sizeof(T));
                std::memcpy(bytes.data(), values.data(), bytes.size());
                return bytes;
            },
            once,
            false,
        };
    }

    // A bound input, written exactly once before execution.
    template<class T>
    Tensor create(const Shape& shape, const std::function<std::vector<T>(std::mt19937&)>& provider) {
        Scope scope(*this);
        auto t = Tensor::empty<T>(shape).input();
        bind<T>(t, provider, /*once=*/true);
        return t;
    }

    // A re-bindable input, rewritten on every (re-)execution — the mechanism
    // behind the loop's per-iteration values.
    template<class T>
    Tensor value(const Shape& shape, const std::function<std::vector<T>(std::mt19937&)>& provider) {
        Scope scope(*this);
        auto t = Tensor::empty<T>(shape).input();
        bind<T>(t, provider);
        return t;
    }

    // Reading back (execution side) — as in the real Context::read<T>.
    template<class T>
    std::vector<T> read(const Tensor& t) const {
        std::vector<T> data(static_cast<std::size_t>(ggml_nelements(*t)));
        ggml_backend_tensor_get(*t, data.data(), 0, ggml_nbytes(*t));
        return data;
    }

    void copy(const Tensor& src, const Tensor& dst) const {
        ggml_backend_tensor_copy(*src, *dst);
    }

    // The one-shot allocation (the model of ggml_backend_alloc_ctx_tensors_
    // from_buft): a single aligned buffer for the whole context — every
    // tensor gets a slot, no liveness (the context lives for the whole
    // generation).
    void allocate(ggml_backend_t backend) {
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_, backend);
        assert(buffer_ != nullptr);
    }

    // The temporary contexts are disposable after their graph executes: give
    // the tensor metadata (and the graph's cgraph) back to the context's
    // pool. The compute buffer is kept by the scheduler.
    void dispose() {
        ggml_reset(ctx_);
        bindings_.clear();
    }

private:
    std::string name_;
    std::vector<std::byte> metadata_;  // the ggml context's mem pool (owned)
    ggml_context* ctx_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    std::size_t capacity_ = 0;
    std::map<ggml_tensor*, Binding> bindings_;
};
// ----------------------------------------------------------------------------
// Tensor creation / ops (the analog of src/ggml/Tensor.cpp): defined after
// Context — they route through the active Scope's context.
// ----------------------------------------------------------------------------

template<class T>
Tensor Tensor::empty(const Shape& shape) {
    return Tensor{ggml_new_tensor(*Scope::context(), DType<T>::value,
                                      static_cast<int>(shape.rank()), shape.data())};
}

Tensor Tensor::input() const { ggml_set_input(t_); return *this; }

Tensor Tensor::operator+(const Tensor& rhs) const { return Tensor{ggml_add(*Scope::context(), t_, *rhs)}; }
Tensor Tensor::operator*(const Tensor& rhs) const { return Tensor{ggml_mul(*Scope::context(), t_, *rhs)}; }
Tensor Tensor::operator-(const Tensor& rhs) const { return Tensor{ggml_sub(*Scope::context(), t_, *rhs)}; }


// ============================================================================
// The monad: Computation<T>, bind, output, repeat, state
// ============================================================================
//
// Computation<T> is the build-phase value: a Tensor (a ggml tensor handle in
// one of the computation's contexts) plus a shared Description of the whole
// computation. The build allocates nothing and executes nothing; the result
// is a lazy description that the Executor (the run phase) plans and executes
// in one pass.
//
// Everything below is domain-independent: nothing in this section knows
// about counters, schedules, or flux.

struct Output {
    std::string name;
    std::optional<Context> context; // the graph's scratch (scheduler-allocated, disposable)
    std::optional<Tensor> output;   // the graph's output (in the temporary context)
    bool loop = false;
    std::size_t count = 0;
    std::size_t iter = 0;           // the loop clock: the per-iteration providers read it
    std::optional<std::pair<Tensor, Tensor>> feedback;  // the loop: next_state -> state cell
    std::vector<std::pair<Tensor, Tensor>> saves;       // boundaries: temporary -> state cell
    ggml_cgraph* graph = nullptr;   // built at plan time (ggml_build_forward_expand)
};

struct Description {
    Context* weights = nullptr;          // provided from outside (pinned)
    std::optional<Context> state;      // created on demand — by state() only
    std::vector<Output> outputs;

    // The state context: the monad's to create and manage. The bodies and
    // the pipeline reach it through state() only — they name the values that
    // cross graph boundaries, not the context those values live in.
    Context& state_ctx() {
        if (!state)
            state.emplace("state");
        return *state;
    }

    Output& add_output(const char* name) {
        outputs.push_back(Output());
        auto& r = outputs.back();
        r.name = name;
        r.context.emplace(std::string("temp: ") + name);
        return r;
    }

    // The output whose temporary context holds the tensor (ggml tensors do
    // not point back at their context, so the lookup walks the outputs).
    Output& output_of(const Tensor& t) {
        for (auto& op : outputs)
            for (ggml_tensor* cur = ggml_get_first_tensor(**op.context); cur != nullptr;
                 cur = ggml_get_next_tensor(**op.context, cur))
                if (cur == *t)
                    return op;
        throw std::runtime_error("output_of: tensor not in any output's temporary context");
    }
};

template<class T>
struct Computation {
    Tensor ref;
    std::shared_ptr<Description> desc;

    Tensor operator *() const {
        return ref;
    }

    Tensor* operator ->() {
        return &ref;
    }
};

// bind — the Kleisli chain (join): the continuation receives the value and
// returns the next computation's output (a U). The description is carried by
// the monad — the body returns the value, not the Computation.
template<class T, class U, class F>
Computation<U> bind(Computation<T> c, F f) {
    return {f(*c), std::move(c.desc)};
}

// output — a single-execution graph: the body (Scope scope, ...) builds the
// low-level ggml tensor chain (Module::forward calls, Tensor ops) in a fresh
// temporary context and returns the graph's output (a T). The Scope is
// created by the monad and is the body's argument, so it is alive for the
// whole body. The description is carried by the monad — the body returns
// the value, not the Computation. The temporary context is disposable after
// the graph executes (the scheduler allocates it); to share the output with
// later outputs, wrap the result in state().
template<class Body>
Computation<Tensor> output(std::shared_ptr<Description> desc, const char* name, Body body) {
    auto& r = desc->add_output(name);
    Scope scope(*r.context);  // the body's argument: alive for the whole body
    auto out = body(scope);
    r.output = out;
    return {out, desc};
}

// repeat — the monadic loop:
//
//   build: the body (Scope scope, Computation<S> state, size_t& iter) is
//          invoked once with the output's Scope (a fresh temporary context —
//          the Scope is the body's argument, alive for the whole body), the
//          state cell, and the loop's iter clock; the Kleisli morphism it
//          records is the loop body as a single graph. The body returns the
//          next state (a S); the description is the monad's to carry.
//   run:   the graph is re-executed `count` times. Before each execution the
//          body's re-bindable (non-once) inputs are rewritten — their
//          providers iterate the per-iteration values (they read the loop's
//          `iter` clock). After each execution the body's output is copied
//          back into the state cell. The value after the loop is the state
//          cell.
template<class S, class Body>
Computation<S> repeat(Computation<S> state, std::size_t count, Body body, const char* name = "loop") {
    auto& r = state.desc->add_output(name);
    r.loop = true;
    r.count = count;
    Scope scope(*r.context);  // the body's argument: alive for the whole body
    auto next = body(scope, state, r.iter);
    r.output = next;
    r.feedback = {next, *state};
    return state;
}

// state — a value crosses a graph boundary. The scheduler's allocation is
// per-graph (not shared between ggml graphs), so a value produced in one
// graph and consumed in a later one must live outside both temporary
// contexts: it is copied into a state context cell right after the
// producing graph executes.
template<class T>
Computation<T> state(Computation<T> t) {
    auto cell = t.desc->state_ctx().tensor(t->dtype(), t->shape());
    cell.name(t->name());
    t.desc->output_of(*t).saves.push_back({*t, cell});
    return {cell, t.desc};
}

// save (provider overload) — a value that must exist from the start, before
// any graph runs: the initial counter cell. It is created in the state
// context — the monad's, created on demand. The caller provides the value,
// not the context: it never has to know when a tensor belongs to the state
// context.
template<class T>
Computation<Tensor> state(std::shared_ptr<Description> desc, const Shape& shape,
                    const std::function<std::vector<T>(std::mt19937&)>& provider, const std::string& name) {
    auto cell = desc->state_ctx().create<T>(shape, provider);
    cell.name(name);
    return {cell, desc};
}

// ============================================================================
// Module stand-ins (the real modules are Module::forward(Scope, ...) methods
// building the low-level ggml tensor chain in the scope's context — that is
// the "working" part the monad does not change; here: two trivial modules)
// ============================================================================

struct Adder {
    Tensor scale_w;  // the weights context (stand-ins for GGUF tensors)
    Tensor bias_w;

    // (a + b) * scale_w + bias_w — with the stand-in weights (1.0, 0.0) this
    // is the minimal 1 + 2.
    Tensor forward(Scope scope, Tensor a, Tensor b) {
        return ((a + b) * scale_w + bias_w).name("adder out");
    }
};

struct CounterStep {
    // The integrate step of the denoising loop (real: x + dt * noise_pred):
    // here, the pure counter step x + step.
    Tensor forward(Scope scope, Tensor state, Tensor step) {
        return (state + step).name("next state");
    }
};

// ============================================================================
// The pipeline (the analog of Flux2KleinPipeline)
// ============================================================================

struct Options {
    int count = 5;  // the number of loop iterations
};

struct Pipeline {
    Adder adder;
    CounterStep counter_step;

    // The monad rework of operator(): instead of building graphs in several
    // contexts and driving per-stage Computation objects (each allocating,
    // planning and executing one graph), this composes the whole
    // "generation" into one lazy description and returns the (unexecuted)
    // computation of the final value.
    Computation<Tensor> compute(Context& weights_ctx, const Options& options) {
        auto desc = std::make_shared<Description>();
        desc->weights = &weights_ctx;

        //
        // 1. 1 + 2: a single-execution output; the output crosses into the
        //    total output (state).
        //
        auto sum = state(output(desc, "adder",
            [&](Scope scope) -> Tensor {
                auto a = scope.context().create<float>(
                    {1}, [](std::mt19937&) { return std::vector<float>{1.0f}; });
                a.name("a (1)");
                auto b = scope.context().create<float>(
                    {1}, [](std::mt19937&) { return std::vector<float>{2.0f}; });
                b.name("b (2)");
                return adder.forward(scope, a, b).name("sum");
            }));

        //
        // 2. The counter's state: a state cell (the loop's "pure"):
        //    the monad creates it in the state context, because it crosses
        //    the loop iterations and the total output. Starts at 0.
        //
        auto counter = state<float>(desc, {1},
            [](std::mt19937&) { return std::vector<float>{0.0f}; }, "counter");

        //
        // 3. Denoise — here a simple counter: x = 0, repeat: x = x + step.
        //    The generic monadic loop: the body is built once (one ggml
        //    graph, its own temporary context); the executor re-executes it
        //    `count` times: the re-bindable `step` input is rewritten on
        //    every iteration (its provider runs through the loop's iter
        //    clock — no graph rebuild), and the state is fed back after
        //    each step.
        //
        counter = repeat(counter, options.count,
            [&](Scope scope, Computation<Tensor> x, std::size_t& iter) -> Tensor {
                (void)iter;  // the loop clock — a real body's providers read it
                // The per-iteration value: a re-bindable input.
                auto step = scope.context().value<float>(
                    {1}, [](std::mt19937&) { return std::vector<float>{1.0f}; });
                step.name("step");
                return counter_step.forward(scope, *x, step);
            },
            "counter");

        //
        // 4. total = counter + sum: both values come from the state context.
        //    The output is the computation's result (saved into the state
        //    context).
        //
        return state(output(desc, "total",
            [&](Scope scope) -> Tensor {
                return (*counter + *sum).name("total");
            }));
    }
};

// The analog of from_pretrained + GGUFLoaderVisitor: the tensors are created
// in the weights context (metadata, no buffer yet) and bound with providers
// streaming them from the (stand-in) GGUF file. The buffer comes from the
// one-shot allocator; the executor streams each binding once.
static void load_weights(Context& ctx, Pipeline& p) {
    auto gguf = [&](const char* name, float value) -> Tensor {
        auto t = ctx.tensor(GGML_TYPE_F32, Shape{1});
        t.name(name);
        // Non-once, like the real GGUF bind — but the executor now writes it
        // exactly once (today: re-streamed by every Computation).
        ctx.bind<float>(t, [value](std::mt19937&) { return std::vector<float>{value}; });
        return t;
    };

    p.adder.scale_w = gguf("adder.scale", 1.0f);
    p.adder.bias_w = gguf("adder.bias", 0.0f);
}

// ============================================================================
// The run phase: the executor (real ggml, one CPU backend, one scheduler)
// ============================================================================
//
// One executor replaces the per-stage Computation objects:
//
//   - one CPU backend (ggml_backend_cpu_init) and one scheduler
//     (ggml_backend_sched_new) for the whole generation — the galloc's
//     compute buffer is persistent and reused across graphs (one buffer for
//     the whole generation).
//   - a one-shot allocation pass for the long-lived contexts (weights +
//     state): ggml_backend_alloc_ctx_tensors — a single aligned buffer per
//     context, the model of ggml_backend_alloc_ctx_tensors_from_buft.
//   - one pass over the outputs: build the graph (ggml_build_forward_expand
//     over the output tensor — the body's tensor chain was already recorded
//     in the temporary context at build time), allocate it
//     (ggml_backend_sched_alloc_graph), write the bindings
//     (ggml_backend_tensor_set), compute (ggml_backend_sched_graph_compute),
//     copy the boundary values (ggml_backend_tensor_copy).
//
//   The loop output is a single allocation with multiple computations: the
//   re-bindable inputs are rewritten before every re-execution, and the
//   state is copied back into the state cell after every iteration.

static std::string format_bytes(std::size_t bytes) {
    if (bytes >= 1024 * 1024) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "%.2f MB", double(bytes) / (1024.0 * 1024.0));
        return buf;
    }
    if (bytes >= 1024)
        return std::to_string(bytes / 1024) + " KB";
    return std::to_string(bytes) + " B";
}

static std::size_t count_tensors(const Context& ctx) {
    std::size_t n = 0;
    for (ggml_tensor* t = ggml_get_first_tensor(*ctx); t != nullptr; t = ggml_get_next_tensor(*ctx, t))
        ++n;
    return n;
}

// TODO: migrate this to ExecutionRuntime?
class Executor {
public:
    explicit Executor(const Computation<Tensor>& result)
        : result_(*result), desc_(result.desc)
    {
        // One CPU backend + one scheduler (reused across all the graphs).
        backend_ = ggml_backend_cpu_init();
        ggml_backend_t backends[1] = {backend_};
        sched_ = ggml_backend_sched_new(backends, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE,
                                        /*parallel=*/false, /*op_offload=*/true);

        // TODO: solve tensor splits here

        // The one-shot allocation point: all the module forwards have been
        // called — all the low-level ggml tensor chains exist. Allocate the
        // long-lived contexts in a single non-incremental pass (one aligned
        // buffer per context).
        if (desc_->weights) {
            desc_->weights->allocate(backend_);
            std::cout << "  [weights] buffer: "
                      << format_bytes(ggml_backend_buffer_get_size(desc_->weights->buffer()))
                      << "  (one-shot, pinned — as today's USAGE_WEIGHTS)\n";
        }
        if (desc_->state) {
            desc_->state->allocate(backend_);
            std::cout << "  [state]   buffer: "
                      << format_bytes(ggml_backend_buffer_get_size(desc_->state->buffer()))
                      << "  (one-shot, lives for the whole generation)\n";
        }
    }

    ~Executor() {
        if (sched_ != nullptr)
            ggml_backend_sched_free(sched_);
        if (backend_ != nullptr)
            ggml_backend_free(backend_);
    }

    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;

    void dump() const;
    void run();
    std::vector<float> read_result() const;
    const Tensor& result() const { return result_; }

private:
    void build_graph(Output& r) const;
    void write_bindings(Context* ctx, bool once_only);
    void copy(const Tensor& src, const Tensor& dst, const char* kind);

    Tensor result_;
    std::shared_ptr<Description> desc_;
    ggml_backend_t backend_ = nullptr;
    ggml_backend_sched_t sched_ = nullptr;
    std::mt19937 rng_{0x5EED};
};

void Executor::build_graph(Output& r) const {
    // The cgraph is built in the temporary context (its memory comes from
    // the context's pool): the body's tensor chain was already recorded
    // there at build time; ggml_build_forward_expand expands it into the
    // graph (the nodes, and the leaves — the bound inputs of this context
    // and the pre-allocated weights / state tensors).
    r.graph = ggml_new_graph_custom(**r.context, r.context->capacity(), /*grads=*/false);
    ggml_set_output(**r.output);
    ggml_build_forward_expand(r.graph, **r.output);
    // Reserve the compute buffer for this graph (it persists on the
    // scheduler and is reused — and grown — for later graphs; in the
    // project this is done once, with a max-size measure graph). In this
    // ggml version reserve_size measures without allocating, so it cannot
    // be the only reserve: alloc_graph would then skip the allocation.
    assert(ggml_backend_sched_reserve(sched_, r.graph));
}

void Executor::write_bindings(Context* ctx, bool once_only) {
    for (auto& [t, b] : ctx->bindings()) {
        if (b.unbound || b.once != once_only)
            continue;
        auto bytes = b.provider(rng_);
        if (bytes.size() != ggml_nbytes(t))
            throw std::runtime_error(std::string("write_bindings: size mismatch for '") + ggml_get_name(t) + "'");
        ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
        std::cout << "     write '" << ggml_get_name(t) << "' " << Tensor{t}.shape().to_string() << " ["
                  << format_bytes(bytes.size()) << "] -> " << ctx->name() << " buffer";
        // Print the value for small float tensors (the inputs, the loop's step).
        const auto n = ggml_nelements(t);
        if (t->type == GGML_TYPE_F32 && n > 0 && n <= 8) {
            const auto* f = reinterpret_cast<const float*>(bytes.data());
            std::cout << "  = [";
            for (int64_t i = 0; i < n; ++i) {
                if (i)
                    std::cout << ", ";
                char bbuf[32];
                std::snprintf(bbuf, sizeof bbuf, "%.3f", f[i]);
                std::cout << bbuf;
            }
            std::cout << "]";
        }
        std::cout << "\n";
    }
}

void Executor::copy(const Tensor& src, const Tensor& dst, const char* kind) {
    // A tensor in a temporary context is copied into the state context
    // (ggml_backend_tensor_copy — the project's Context::copy).
    ggml_backend_tensor_copy(*src, *dst);
    std::cout << "     copy '" << src.name() << "' -> '" << dst.name() << "'  [" << kind << "]\n";
}

void Executor::dump() const {
    const auto& d = *desc_;
    std::cout << "\n=== PLAN ===\n";

    std::cout << "the description (built by the monad — nothing allocated or executed yet):\n";
    for (std::size_t i = 0; i < d.outputs.size(); ++i) {
        const auto& r = d.outputs[i];
        std::cout << "  " << i << ": " << std::left << std::setw(14) << r.name
                  << (r.loop ? std::string("[loop x") + std::to_string(r.count) + "]" : "[once]")
                  << std::right << "  " << count_tensors(*r.context) << " tensors in the temporary context";
        if (r.loop)
            std::cout << "\n                            feedback '" << r.feedback->first.name()
                      << "' -> '" << r.feedback->second.name() << "' after each iteration";
        std::cout << "\n";
    }

    auto print_ctx = [&](const Context* ctx, const char* label) {
        std::cout << "\n  the " << label << " buffer (one-shot, "
                  << format_bytes(ggml_backend_buffer_get_size(ctx->buffer())) << "):\n";
        for (ggml_tensor* t = ggml_get_first_tensor(**ctx); t != nullptr; t = ggml_get_next_tensor(**ctx, t))
            std::cout << "     " << std::left << std::setw(16) << ggml_get_name(t) << std::right
                      << "  " << std::setw(8) << Tensor{t}.shape().to_string()
                      << "  " << format_bytes(ggml_nbytes(t)) << "\n";
    };

    if (d.weights)
        print_ctx(d.weights, "weights");
    if (d.state)
        print_ctx(&d.state.value(), "state");

    std::cout << "\n  the compute buffer (the scheduler's galloc — sized per graph at alloc\n"
              << "  time, reused across graphs): see the RUN below\n";
}

void Executor::run() {
    std::cout << "\n=== RUN ===\n";
    std::cout << "one CPU backend + one scheduler (the compute buffer is reused across graphs)\n";
    std::cout << "one pass over the outputs (build order == execution order)\n\n";

    // The weights: the GGUF bindings are streamed once into the weights
    // buffer. (Today: re-streamed by every Computation.)
    if (desc_->weights) {
        std::cout << "  [weights] " << count_tensors(*desc_->weights) << " tensors\n";
        write_bindings(desc_->weights, /*once_only=*/false);
    }

    // The state's one-shot bindings (the initial counter).
    if (desc_->state) {
        std::cout << "  [state]\n";
        write_bindings(&desc_->state.value(), /*once_only=*/true);
    }
    std::cout << "\n";

    for (std::size_t i = 0; i < desc_->outputs.size(); ++i) {
        auto& r = desc_->outputs[i];
        std::cout << "  -- output " << i << "/" << desc_->outputs.size() - 1 << ": " << r.name;
        if (r.loop)
            std::cout << "  (loop x" << r.count << ")";
        std::cout << "\n";

        // Build the graph over the output tensor (the body's tensor chain in
        // the temporary context, already recorded at build time).
        build_graph(r);
        std::cout << "     graph: " << ggml_graph_n_nodes(r.graph) << " nodes\n";

        // Allocate: the galloc plans the graph's temporary tensors with
        // liveness (the compute buffer is sized to the high-water mark of
        // the live set); the weights / state tensors are already allocated
        // and skipped. build_graph reserved the buffer; alloc_graph assigns
        // the tensors' addresses in it.
        std::cout << "     alloc: compute buffer "
                  << format_bytes(ggml_backend_sched_get_buffer_size(sched_, backend_))
                  << "  (persistent, reused)\n";
        if (!ggml_backend_sched_alloc_graph(sched_, r.graph))
            throw std::runtime_error("ggml_backend_sched_alloc_graph failed: " + r.name);

        // The bound inputs (once): written into the freshly allocated graph
        // tensors (ggml_backend_tensor_set).
        write_bindings(&r.context.value(), /*once_only=*/true);

        if (r.loop) {
            // The graph is a single allocation with multiple computations.
            for (r.iter = 0; r.iter < r.count; ++r.iter) {
                std::cout << "     iter " << r.iter + 1 << "/" << r.count << ":\n";
                // The re-bindable inputs: rewritten on every (re-)execution —
                // their providers run through the loop's iter clock.
                write_bindings(&r.context.value(), /*once_only=*/false);
                if (ggml_backend_sched_graph_compute(sched_, r.graph) != GGML_STATUS_SUCCESS)
                    throw std::runtime_error("ggml_backend_sched_graph_compute failed: " + r.name);
                // The feedback: the next state is written back into the
                // state cell (on every iteration, including the last — the
                // cell must hold the final state; today's loop never does
                // this, which is why its result is broken).
                copy(r.feedback->first, r.feedback->second, "feedback");
            }
        } else {
            if (ggml_backend_sched_graph_compute(sched_, r.graph) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("ggml_backend_sched_graph_compute failed: " + r.name);
        }

        for (auto& [s, dst] : r.saves)
            copy(s, dst, "boundary");

        // The temporary context is disposable: the graph has been computed,
        // the state is across the boundary. (The compute buffer is kept by
        // the scheduler.)
        r.context.reset();
        std::cout << "     dispose temporary context '" << r.context->name() << "'\n\n";
    }
}

std::vector<float> Executor::read_result() const {
    // The result is a state cell (the pipeline returns state(...)): the state
    // context is alive for the whole generation, so it can be read back.
    assert(desc_->state);
    auto data = desc_->state->read<float>(result_);
    std::cout << "\n  read '" << result_.name() << "' " << result_.shape().to_string() << " f32 ["
              << format_bytes(result_.nbytes()) << "] from the '" << desc_->state->name() << "' buffer\n";
    return data;
}

// ============================================================================
// main
// ============================================================================

static std::optional<Tensor> find_tensor(const Context& ctx, const char* name) {
    for (ggml_tensor* t = ggml_get_first_tensor(*ctx); t != nullptr; t = ggml_get_next_tensor(*ctx, t))
        if (std::strcmp(ggml_get_name(t), name) == 0)
            return Tensor{t};
    return std::nullopt;
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);  // unbuffered: trace survives a crash
    // The weights context: provided from outside (from_pretrained loads the
    // modules into it). Pinned: allocated once, never freed.
    Context weights_ctx("weights");
    Pipeline pipeline;
    load_weights(weights_ctx, pipeline);

    Options options;  // 5 loop iterations

    // TODO: create parent scope with ShardingRuntime here (it's going to be used during compute())

    // 1. Build — the monadic description of the whole "generation".
    //    No allocation, no execution.
    auto image = pipeline.compute(weights_ctx, options);

    // 2. Plan + allocate — one-shot, now that all the module forwards have
    //    been called (all the low-level ggml tensor chains exist).
    Executor executable(image);
    executable.dump();

    // 3. Run — one pass over the outputs.
    executable.run();

    // 4. Read the final result from the state context.
    auto total = executable.read_result();
    std::cout << "  -> total = " << total.front() << " (as expected 8 = 5 + 3)\n";

    // 5. Verify — the values in the state context: the executor ran the
    //    graphs on real ggml (CPU).
    auto& state_ctx = image.desc->state_ctx();
    bool ok = true;
    const char* cells[] = {"sum", "counter", "total"};
    const float expected[] = {3.0f, 5.0f, 8.0f};
    std::cout << "\n=== CHECK ===\n";
    for (int i = 0; i < 3; ++i) {
        auto t = find_tensor(state_ctx, cells[i]);
        if (!t) {
            std::cout << "  FAIL: state cell '" << cells[i] << "' not found\n";
            ok = false;
            continue;
        }
        const float got = state_ctx.read<float>(*t).front();
        const bool pass = got == expected[i];
        std::cout << "  " << (pass ? "PASS" : "FAIL") << ": " << cells[i]
                  << " = " << got << " (expected " << expected[i] << ")\n";
        ok = ok && pass;
    }

    std::cout << (ok ? "\nOK — the monad executed on real ggml (single CPU backend)\n"
                     : "\nFAILED\n");
    return ok ? 0 : 1;
}
