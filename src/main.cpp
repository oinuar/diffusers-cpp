// ============================================================================
// PoC: the computational monad for Flux2KleinPipeline
// ============================================================================
//
// A standalone model of the reworked Flux2KleinPipeline (src/diffusers/
// pipelines/flux2/Flux2KleinPipeline.cpp). The Tensor / Context / Scope
// stand-ins mirror the API surface of ggml::Tensor / ggml::Context /
// ggml::Scope; the modules mirror the Module::forward(Scope, ...) methods;
// the pipeline composition mirrors operator(). The point is the layer in
// between — the computational monad — which this PoC is meant to migrate
// into the project as the replacement of src/ggml/Computation.hpp.
//
// Two layers:
//
//   low level (unchanged)  Module::forward() / Tensor ops build the low-level
//                          ggml tensor chain in a context (a Scope). The
//                          pipeline's make_*_graph code becomes the region
//                          bodies below, calling the same forwards.
//
//   high level (the monad) Computation<T> is the build-phase value: a Tensor
//                          plus a shared Description of the whole
//                          computation. bind threads values; region()
//                          records a single-execution graph; fold() records
//                          a loop (the body is built once, re-executed N
//                          times); save() puts a value in the state context
//                          (a copy of a graph's output after the graph
//                          executes, or a state cell created up front from a
//                          provider). The bodies receive the graph's Scope
//                          and return the value — the description is the
//                          monad's to carry.
//
// Memory model — three context classes:
//
//   weights    provided from outside (from_pretrained), kept alive / pinned.
//              Allocated once, never freed (as today: USAGE_WEIGHTS).
//   state      created by save() when a value must share data between
//              ggml graphs: the values that cross graph boundaries (image
//              latents, prompt embeds, the denoising state cell, the final
//              image). Allocated once, lives for the whole generation.
//   temporary  one per graph: the graph's working tensors. Allocated by the
//              scheduler (galloc liveness), disposable after the graph
//              executes.
//
//   The state context exists because the scheduler's allocation is per
//   graph (it is not shared between ggml graphs): a value produced by one
//   graph and consumed by a later one must live outside both temporary
//   contexts.
//
// Allocation — one-shot, at the point where all Module::forward() calls have
// completed (all low-level ggml tensor chains exist):
//
//   - long-lived contexts (weights + state): a single non-incremental pass
//     of the simplified allocator (plain aligned bump per context — the
//     model of ggml_backend_alloc_ctx_tensors_from_buft, which skips views
//     and already-allocated tensors).
//   - temporary contexts: the scheduler, per graph (the compute buffer is
//     sized to the high-water mark of the live set and is reused across
//     graphs — one buffer for the whole generation).
//
// What this replaces in the current code:
//
//   - the four per-stage Computation objects (each: allocate + plan +
//     execute one graph) -> one Executor: one allocation pass, one execution
//     pass over the regions.
//   - the imperative denoise loop (re-executing the graph with the state
//     feedback missing — the latents input is never updated) -> fold: the
//     per-step timestep/dt are re-bindable inputs (their providers iterate
//     the schedule through the loop's iter clock), and the state is fed
//     back after each step (an explicit copy into the state cell).
//   - the incremental allocator (use/allocate per Computation, the weights
//     re-streamed per graph) -> the one-shot non-incremental allocator
//     (the weights are streamed once).
//
// What this enables for src/ggml:
//
//   - Graph: loses the scheduler planning and the binding collection (the
//     Executor drives the scheduler); it becomes (context + outputs + the
//     ggml_cgraph build).
//   - Computation: the Computation<T> monad value (build) + the Executor
//     (run).
//   - Allocator: a single non-incremental pass.
//   - Scheduler: the same ggml_backend_sched, driven per region.
//   - Context: unchanged (create/value/bind/read/write/copy) — it also hosts
//     the state context.
//
// Mapping (PoC -> project):
//
//   Tensor/Shape/DType/Context/Scope  ggml::Tensor / Shape / DType / Context
//                                     / Scope (the migration target of the
//                                     stand-ins in this file)
//   Vae/TextEncoder/Transformer       the nn::Module trees (AutoencoderKL-
//                                     Flux2, Qwen3ForCausalLM,
//                                     Flux2Transformer2DModel)
//   Pipeline::compute                 Flux2KleinPipeline::operator()
//   load_weights                      from_pretrained + GGUFLoaderVisitor
//   make_schedule/Schedule            FlowMatchEulerDiscreteScheduler::
//                                     schedule
//   the ids helpers                   prepare_img_ids / prepare_txt_ids /
//                                     prepare_ref_image_ids
//   the pack/unpack helpers           the pack/unpack/patchify/unpatchify
//                                     statics
//   the one-shot allocator            the simplified ggml::Allocator
//   the scheduler                     ggml_backend_sched (galloc model)
//   the executor                      the replacement of the Computation class
//
// Build & run (standalone):
//   g++ -std=c++17 -O2 -Wall -Wextra -o /tmp/poc src/main.cpp && /tmp/poc

#include <algorithm>
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
// Tensor stand-in (mirrors the API shape of src/ggml/Tensor.hpp)
// ============================================================================

class Context;
struct Region;  // the monad's region (defined below) — a temporary context points back to its region
class Tensor;
static Tensor buffer_owner(const Tensor& t);  // resolves a view chain to its buffer-owning tensor

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
    int64_t& operator[](int64_t i) {
        assert(i >= 0 && i < rank_);
        return ne_[rank_ - 1 - i];
    }

    int64_t rank() const { return rank_; }

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
    std::array<int64_t, 4> ne_{};  // ne[0] = the fastest (innermost) dimension
    int64_t rank_ = 0;
};

enum class DType { F32, I32, U8 };

inline std::size_t dtype_size(DType t) {
    switch (t) {
        case DType::F32: return 4;
        case DType::I32: return 4;
        case DType::U8:  return 1;
    }
    return 0;
}

// The tensor metadata (the stand-in for what ggml stores per tensor).
struct TensorMeta {
    Shape shape;
    DType dtype = DType::F32;
    std::string name;
    bool contiguous = true;
    int base = -1;              // view: the tensor this shares a buffer with (-1 = none)
    Context* base_ctx = nullptr; // view: the context that tensor lives in (may differ from this one)
    int producer = -1;          // the node producing this (-1 = input / view)
    std::vector<int> consumers; // the nodes consuming this (through views)
    std::size_t offset = 0;     // slot in the context's buffer (if any)
    bool allocated = false;
    std::size_t size_bytes = 0;
};

struct Tensor {
    Context* ctx = nullptr;
    int id = -1;

    Tensor() = default;
    Tensor(Context* c, int i) : ctx(c), id(i) {}

    bool valid() const { return ctx != nullptr && id >= 0; }
    bool operator==(const Tensor& o) const { return ctx == o.ctx && id == o.id; }
    bool operator!=(const Tensor& o) const { return !(*this == o); }

    Context& context() const;
    const TensorMeta& meta() const;
    TensorMeta& meta();
    const Shape& shape() const;
    DType dtype() const;
    const std::string& name() const;
    const Tensor& name(const std::string& n) const;  // fluent, as in ggml_set_name (sets the ggml name in place)
    int64_t numel() const;
    std::size_t nbytes() const;
    bool is_contiguous() const;

    // Tensor ops (as in the real Tensor): each records a node in the active
    // scope's context. Defined after Context.
    Tensor contiguous() const;
    Tensor scale(float) const;
    Tensor permute(const Shape& order) const;
    Tensor reshape(const Shape& shape) const;
    Tensor narrow(int axis, int64_t start, int64_t len) const;
    Tensor operator*(Tensor rhs) const;
    Tensor operator*(float v) const;
    Tensor operator+(float v) const;
    Tensor operator+(Tensor rhs) const;
    Tensor operator-(Tensor rhs) const;
    Tensor operator/(float v) const;
    Tensor operator/(Tensor rhs) const;

    static Tensor cat(const std::vector<Tensor>& ts, int axis);
    static Tensor stack(const std::vector<Tensor>& ts, int axis);
};

Tensor sqrt(const Tensor& t);

struct Node {
    std::string op;
    int output = -1;
    std::vector<Tensor> inputs;
};

// The Scope stand-in (ggml::Scope): the thread-local "current context" (the
// real Scope also carries the Runtime; here the ops route through the
// context directly).
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

template<class T>
struct DTypeOf;
template<>
struct DTypeOf<float> { static constexpr DType value = DType::F32; };
template<>
struct DTypeOf<int32_t> { static constexpr DType value = DType::I32; };

// The Context stand-in (mirrors src/ggml/Context.hpp): create/value (bound
// inputs), tensor (bare metadata), node (graph ops), read (execution side).
class Context {
public:
    using Provider = std::function<std::vector<std::byte>(std::mt19937&)>;

    struct Binding {
        Provider provider;
        bool once;
        bool unbound = false;
    };

    explicit Context(std::string name) : name_(std::move(name)) {}

    const std::string& name() const { return name_; }
    TensorMeta& meta(int id) { return tensors_[id]; }
    const TensorMeta& meta(int id) const { return tensors_[id]; }
    std::vector<TensorMeta>& tensors() { return tensors_; }
    const std::vector<TensorMeta>& tensors() const { return tensors_; }
    const std::vector<Node>& nodes() const { return nodes_; }
    const std::map<int, Binding>& bindings() const { return bindings_; }
    bool has_tensors() const { return !tensors_.empty(); }
    bool has_buffer() const { return !buffer.empty(); }

    // Tensor creation — metadata only. Allocation happens later: one-shot
    // for the long-lived contexts, per-graph by the scheduler for the
    // temporary ones.
    Tensor tensor(Shape shape, DType dtype) {  // by value: see node()
        tensors_.emplace_back();
        auto& m = tensors_.back();
        m.shape = shape;
        m.dtype = dtype;
        m.size_bytes = static_cast<std::size_t>(shape.numel()) * dtype_size(dtype);
        return Tensor{this, static_cast<int>(tensors_.size() - 1)};
    }

    // Binding (as in the real Context::bind<T>): the provider is called at
    // execution time and the value is (simulated to be) memcpy'd into the
    // tensor's slot.
    void bind_byte(Tensor t, const Provider& provider, bool once = false) {
        bindings_[t.id] = {provider, once, false};
    }

    template<class T>
    void bind(Tensor t, const std::function<std::vector<T>(std::mt19937&)>& provider, bool once = false) {
        bind_byte(t, [provider](std::mt19937& rng) {
            auto values = provider(rng);
            if constexpr (std::is_same_v<T, std::byte>)
                return std::move(values);
            std::vector<std::byte> bytes(values.size() * sizeof(T));
            std::memcpy(bytes.data(), values.data(), bytes.size());
            return bytes;
        }, once);
    }

    // A bound input, written exactly once before execution.
    template<class T>
    Tensor create(const Shape& shape, const std::function<std::vector<T>(std::mt19937&)>& provider) {
        auto t = tensor(shape, DTypeOf<T>::value);
        bind<T>(t, provider, /*once=*/true);
        return t;
    }

    // A re-bindable input, rewritten on every (re-)execution — the mechanism
    // behind the loop's per-iteration values.
    template<class T>
    Tensor value(const Shape& shape, const std::function<std::vector<T>(std::mt19937&)>& provider) {
        auto t = tensor(shape, DTypeOf<T>::value);
        bind<T>(t, provider, /*once=*/false);
        return t;
    }

    // Records a node (an op) — the low-level ggml graph building. Shape is taken by
    // value: callers may pass a reference into this context's own tensors (e.g.
    // meta().shape), which the emplace_back below may reallocate.
    Tensor node(const char* op, Shape shape, DType dtype, std::vector<Tensor> inputs) {
        auto out = tensor(shape, dtype);
        meta(out.id).producer = static_cast<int>(nodes_.size());
        for (auto& in : inputs) {
            // Resolve through the view chain (it may cross contexts) to the
            // tensor that owns the buffer, and track its liveness here only if
            // that tensor lives in this (temporary) context.
            auto owner = buffer_owner(in);
            if (owner.ctx == this)
                owner.meta().consumers.push_back(static_cast<int>(nodes_.size()));
        }
        Node n;
        n.op = op;
        n.output = out.id;
        n.inputs = std::move(inputs);
        nodes_.push_back(std::move(n));
        return out;
    }

    // Reading back (execution side) — as in the real Context::read<T>.
    template<class T>
    std::vector<T> read(const Tensor& t) const {
        const auto& m = meta(t.id);
        assert(has_buffer() && m.allocated && m.size_bytes % sizeof(T) == 0);
        std::vector<T> data(m.size_bytes / sizeof(T));
        std::memcpy(data.data(), buffer.data() + m.offset, m.size_bytes);
        return data;
    }

    // Simulated storage: the long-lived contexts get a plain bump buffer
    // from the one-shot allocator. The temporary contexts get nothing here —
    // the scheduler owns their memory (the compute buffer).
    std::vector<std::byte> buffer;
    Region* region = nullptr;  // back-pointer (temporary contexts only)

private:

    std::string name_;
    std::vector<TensorMeta> tensors_;
    std::vector<Node> nodes_;
    std::map<int, Binding> bindings_;
};

// ----------------------------------------------------------------------------
// Tensor accessors + ops (the analog of src/ggml/Tensor.cpp)
// ----------------------------------------------------------------------------

Context& Tensor::context() const { return *ctx; }
const TensorMeta& Tensor::meta() const { return ctx->meta(id); }
TensorMeta& Tensor::meta() { return ctx->meta(id); }
const Shape& Tensor::shape() const { return meta().shape; }
DType Tensor::dtype() const { return meta().dtype; }
const std::string& Tensor::name() const { return meta().name; }
const Tensor& Tensor::name(const std::string& n) const { ctx->meta(id).name = n; return *this; }
int64_t Tensor::numel() const { return meta().shape.numel(); }
std::size_t Tensor::nbytes() const { return meta().size_bytes; }
bool Tensor::is_contiguous() const { return meta().contiguous; }

// The monad never changes this layer: the region bodies call the same ops
// and module forwards, only the context is the region's temporary context.

static Shape broadcast_shape(const Shape& a, const Shape& b) {
    // Real: Shape::broadcast. PoC: the binary ops take the broadcast
    // superset first (the same convention as the real Tensor ops).
    if (a == b)
        return a;
    if (b.rank() <= 1)
        return a;
    if (a.rank() <= 1)
        return b;
    return a;
}

Tensor Tensor::contiguous() const {
    if (meta().contiguous)
        return *this;
    return Scope::context().node("ggml_contiguous", meta().shape, meta().dtype, {*this});
}

Tensor Tensor::scale(float v) const {
    // Real: ggml_mul with a scalar tensor; the PoC bakes the value into the
    // op name (as with add_scalar) so the traces show the scale.
    char op[48];
    std::snprintf(op, sizeof op, "ggml_scale(%.9g)", v);
    return Scope::context().node(op, meta().shape, meta().dtype, {*this});
}

Tensor Tensor::operator*(Tensor rhs) const {
    return Scope::context().node("ggml_mul", broadcast_shape(shape(), rhs.shape()), dtype(), {*this, rhs});
}

Tensor Tensor::operator*(float v) const { return scale(v); }

Tensor Tensor::operator+(float v) const {
    // Real: the scalar is built as a (1,) tensor and broadcast; the PoC
    // bakes it into the op name.
    char op[48];
    std::snprintf(op, sizeof op, "ggml_add_scalar(%.9g)", v);
    return Scope::context().node(op, meta().shape, meta().dtype, {*this});
}


Tensor Tensor::operator+(Tensor rhs) const {
    return Scope::context().node("ggml_add", broadcast_shape(shape(), rhs.shape()), dtype(), {*this, rhs});
}

Tensor Tensor::operator-(Tensor rhs) const {
    return Scope::context().node("ggml_sub", broadcast_shape(shape(), rhs.shape()), dtype(), {*this, rhs});
}

Tensor Tensor::operator/(float v) const { return scale(1.0f / v); }

Tensor Tensor::operator/(Tensor rhs) const {
    return Scope::context().node("ggml_div", broadcast_shape(shape(), rhs.shape()), dtype(), {*this, rhs});
}

Tensor sqrt(const Tensor& t) {
    return Scope::context().node("ggml_sqrt", t.shape(), t.dtype(), {t});
}

static Tensor view_of(const Tensor& t, const Shape& shape, bool contiguous) {
    // A view (as in GGML_OP_PERMUTE / RESHAPE / VIEW): no node, no buffer of
    // its own — it shares the base's buffer. As in ggml, the view's metadata
    // lives in the active scope's context; the base may live in a different
    // context (weights / state / a temporary).
    auto out = Scope::context().tensor(shape, t.dtype());
    auto& m = out.meta();
    m.base = t.id;
    m.base_ctx = t.ctx;
    m.contiguous = contiguous && t.is_contiguous();
    if (!t.meta().name.empty())
        m.name = t.meta().name;  // views are readable as their base in the traces
    return out;
}

static Tensor buffer_owner(const Tensor& t) {
    Tensor cur = t;
    while (cur.meta().base >= 0)
        cur = Tensor{cur.meta().base_ctx, cur.meta().base};
    return cur;
}

Tensor Tensor::permute(const Shape& order) const {
    assert(order.rank() == meta().shape.rank());
    Shape out(meta().shape.rank());
    for (int i = 0; i < order.rank(); ++i)
        out[i] = meta().shape[order[i]];
    return view_of(*this, out, /*contiguous=*/false);
}

Tensor Tensor::reshape(const Shape& shape) const {
    if (!meta().contiguous)
        // Real: a non-contiguous reshape is first made contiguous (a copy).
        return contiguous().reshape(shape);
    int64_t known = 1;
    int minus = -1;
    for (int i = 0; i < shape.rank(); ++i) {
        const auto v = shape[i];
        if (v == -1) {
            assert(minus < 0);
            minus = i;
        } else
            known *= v;
    }
    Shape out = shape;
    if (minus >= 0) {
        assert(known > 0);
        out[minus] = meta().shape.numel() / known;
    }
    assert(out.numel() == meta().shape.numel());
    return view_of(*this, out, /*contiguous=*/true);
}

Tensor Tensor::narrow(int axis, int64_t, int64_t len) const {
    Shape out = meta().shape;
    out[axis] = len;
    return view_of(*this, out, meta().contiguous);
}

Tensor Tensor::cat(const std::vector<Tensor>& ts, int axis) {
    assert(!ts.empty());
    Shape out = ts[0].shape();
    int64_t total = 0;
    for (auto& t : ts)
        total += t.shape()[axis];
    out[axis] = total;
    return Scope::context().node("ggml_concat", out, ts[0].dtype(), ts);
}

Tensor Tensor::stack(const std::vector<Tensor>& ts, int axis) {
    assert(!ts.empty());
    const auto& in = ts[0].shape();
    Shape out(in.rank() + 1);
    // insert a new dimension of size ts.size() at `axis`
    for (int i = 0; i <= in.rank(); ++i)
        out[i] = (i == axis) ? static_cast<int64_t>(ts.size())
                             : in[i - (i > axis ? 1 : 0)];
    return Scope::context().node("ggml_stack", out, ts[0].dtype(), ts);
}

// ============================================================================
// The monad: Computation<T>, bind, region, fold, save
// ============================================================================
//
// Computation<T> is the build-phase value: a Tensor (a tensor handle in one
// of the computation's contexts) plus a shared Description of the whole
// computation. The build allocates nothing and executes nothing; the result
// is a lazy description that the Executor (the run phase) plans and
// executes in one pass.
//
//   bind    the Kleisli chain (join): the continuation receives the value
//           and returns the next computation's output
//   region  a single-execution graph: the body (Scope, ...) builds the
//           low-level tensor chain in a fresh temporary context and returns
//           the graph's output
//   save    the door to the state context: save(value) copies a graph's
//           output into a state cell right after the graph executes;
//           save(desc, shape, provider) creates a state cell up front
//   fold    the generic monadic loop: the body (Scope, state, iter) is
//           built once (one ggml graph), re-executed N times, and the state
//           is fed back after each iteration
//
// Everything above is domain-independent: nothing in this section knows
// about denoising, schedules, or flux.

struct Region {
    std::string name;
    std::unique_ptr<Context> temp;  // the graph's scratch (scheduler-allocated, disposable)
    std::optional<Tensor> output;   // the graph's output (in the temporary context)
    bool loop = false;
    std::size_t count = 0;
    std::size_t iter = 0;           // the loop clock: the per-iteration providers read it
    std::optional<std::pair<Tensor, Tensor>> feedback;  // the loop: next_state -> state cell
    std::vector<std::pair<Tensor, Tensor>> saves;       // boundaries: temporary -> state cell
    std::size_t high_water = 0;     // set by the scheduler's alloc (galloc model)
};

struct Description {
    Context* weights = nullptr;          // provided from outside (pinned)
    std::unique_ptr<Context> state;      // created on demand — by save() only
    std::vector<std::unique_ptr<Region>> regions;

    // The state context: the monad's to create and manage. The bodies and
    // the pipeline reach it through save() only — they name the values that
    // cross graph boundaries, not the context those values live in.
    Context& state_ctx() {
        if (!state)
            state = std::make_unique<Context>("state");
        return *state;
    }

    Region& add_region(const char* name) {
        regions.push_back(std::make_unique<Region>());
        auto& r = *regions.back();
        r.name = name;
        r.temp = std::make_unique<Context>(std::string("temp: ") + name);
        r.temp->region = &r;
        return r;
    }

    Region& region_of(const Tensor& t) {
        assert(t.ctx && t.ctx->region);
        return *t.ctx->region;
    }
};

template<class T>
struct Computation {
    Tensor ref;
    std::shared_ptr<Description> desc;
};

// bind — the Kleisli chain (join): the continuation receives the value and
// returns the next computation's output (a U). The description is carried by
// the monad — the body returns the value, not the Computation.
template<class T, class U, class F>
Computation<U> bind(Computation<T> c, F f) {
    return {f(c.ref), std::move(c.desc)};
}

// region — a single-execution graph: the body (Scope scope, ...) builds the
// low-level tensor chain (Module::forward calls, Tensor ops) in a fresh
// temporary context and returns the graph's output (a T). The Scope is
// created by the monad and is the body's argument, so it is alive for the
// whole body. The description is carried by the monad — the body returns
// the value, not the Computation. The temporary context is disposable after
// the graph executes (the scheduler allocates it); to share the output with
// later regions, wrap the result in save().
template<class T, class Body>
Computation<T> region(std::shared_ptr<Description> desc, const char* name, Body body) {
    auto& r = desc->add_region(name);
    Scope scope(*r.temp);  // the body's argument: alive for the whole body
    auto out = body(scope);
    r.output = out;
    return {out, desc};
}

// fold — the monadic loop (generic — no domain knowledge):
//
//   build: the body (Scope scope, Computation<S> state, size_t& iter) is
//          invoked once with the region's Scope (a fresh temporary context —
//          the Scope is the body's argument, alive for the whole body), the
//          state cell, and the loop's iter clock; the Kleisli morphism it
//          records is the loop body as a single graph. The body returns the
//          next state (a S); the description is the monad's to carry.
//   run:   the graph is re-executed `count` times. Before each execution the
//          body's re-bindable (non-once) inputs are rewritten — their
//          providers iterate the per-iteration values (they read the loop's
//          `iter` clock). After each execution the body's output is copied
//          back into the state cell — the feedback that today's re-executed
//          denoise graph is missing. The value after the loop is the state
//          cell.
template<class S, class Body>
Computation<S> fold(Computation<S> state, std::size_t count, Body body, const char* name = "fold") {
    auto& r = state.desc->add_region(name);
    r.loop = true;
    r.count = count;
    Scope scope(*r.temp);  // the body's argument: alive for the whole body
    auto next = body(scope, state, r.iter);
    r.output = next;
    r.feedback = {next, state.ref};
    return state;
}

// save — a value crosses a graph boundary. The scheduler's allocation is
// per-graph (not shared between ggml graphs), so a value produced in one
// graph and consumed in a later one must live outside both temporary
// contexts: it is copied into a state context cell right after the
// producing graph executes.
template<class T>
Computation<T> save(Computation<T> t) {
    auto cell = t.desc->state_ctx().tensor(t.ref.shape(), t.ref.dtype());
    cell.name(t.ref.name());
    t.desc->region_of(t.ref).saves.push_back({t.ref, cell});
    return {cell, t.desc};
}

// save (provider overload) — a value that must exist from the start, before
// any graph runs: the initial latents, the denoising state cell. It is
// created in the state context — the monad's, created on demand. The caller
// provides the value, not the context: it never has to know when a tensor
// belongs to the state context.
template<class T>
Computation<Tensor> save(std::shared_ptr<Description> desc, const Shape& shape,
                    const std::function<std::vector<T>(std::mt19937&)>& provider, const std::string& name) {
    auto cell = desc->state_ctx().create<T>(shape, provider);
    cell.name(name);
    return {cell, desc};
}

// ============================================================================
// Domain stand-ins (mirrors of the pipeline's host-side types)
// ============================================================================

struct Image {
    int width = 0;
    int height = 0;
    std::vector<std::byte> pixels;  // HWC, uint8
};

struct Options {
    int height = 1024;
    int width = 1024;
    int num_inference_steps = 8;
    int num_images_per_prompt = 1;
    std::optional<std::vector<float>> init_latents;
};

// Mirrors diffusers/schedulers/Schedule (timesteps + n+1 sigmas).
class Schedule {
public:
    struct Step {
        float timestep;
        float sigma;
        float sigma_next;
        float dt;
    };

    Schedule(std::vector<float> timesteps, std::vector<float> sigmas)
        : timesteps_(std::move(timesteps)), sigmas_(std::move(sigmas)) {
        assert(sigmas_.size() == timesteps_.size() + 1);
    }

    std::size_t size() const { return timesteps_.size(); }
    Step operator[](std::size_t i) const {
        return {timesteps_[i], sigmas_[i], sigmas_[i + 1], sigmas_[i + 1] - sigmas_[i]};
    }

private:
    std::vector<float> timesteps_;
    std::vector<float> sigmas_;
};

// ============================================================================
// Module stand-ins
// ============================================================================
//
// The real modules are Module::forward(Scope, ...) methods building the
// low-level ggml tensor chain in the scope's context (Tensor ops,
// sub-modules). That is the "working" part — the monad does not change it.
// Here: the same shape — forward(Scope, ...) recording nodes and returning
// the output Tensor. The region bodies below call them exactly as
// Flux2KleinPipeline::operator() calls the real module forwards, only the
// context is the region's temporary context.

struct Vae {
    // Weights (created in the weights context by load_weights — stand-ins
    // for the VAE's GGUF tensors).
    Tensor enc0, enc1;
    Tensor dec0, dec1, dec2;
    Tensor bn_mean, bn_var;
    int scale_factor = 8;
    int latent_channels = 16;
    float batch_norm_eps = 1e-5f;

    // The analog of AutoencoderKLFlux2::encode + dist.mode()
    Tensor encode(Scope scope, Tensor img) {
        auto& ctx = scope.context();
        const auto H = img.shape()[2];
        const auto W = img.shape()[3];
        auto h = ctx.node("ggml_conv_2d", {1, 32, H / 8, W / 8}, DType::F32, {img, enc0})
                     .name("vae encoder conv");
        return ctx.node("dist mode", {1, latent_channels, H / 8, W / 8}, DType::F32, {h})
                   .name("vae encode mode");
    }

    // The analog of AutoencoderKLFlux2::decode
    Tensor decode(Scope scope, Tensor z) {
        auto& ctx = scope.context();
        const auto H = z.shape()[2];
        const auto W = z.shape()[3];
        auto h = ctx.node("ggml_conv_2d", {1, 32, H * 2, W * 2}, DType::F32, {z, dec0})
                     .name("vae decoder conv 0");
        h = ctx.node("ggml_conv_2d", {1, 16, H * 4, W * 4}, DType::F32, {h, dec1})
                .name("vae decoder conv 1");
        return ctx.node("ggml_conv_2d", {1, 3, H * 8, W * 8}, DType::F32, {h, dec2})
                   .name("vae decoder conv 2");
    }
};

struct TextEncoder {
    Tensor embed_w;
    std::vector<Tensor> layer_w;
    int hidden = 128;
    std::vector<int> extract = {0, 2, 3};  // real: layers 9/18/27 of 36

    // The analog of Qwen3ForCausalLM::forward — the extracted hidden states
    // come out through the hidden_states parameter (as in the real one).
    void forward(Scope scope, Tensor input_ids, Tensor attention_mask, std::vector<Tensor>* hidden_states) {
        auto& ctx = scope.context();
        const auto B = input_ids.shape()[0];
        const auto L = input_ids.shape()[1];
        auto h = ctx.node("ggml_embedding", {B, L, hidden}, DType::F32, {input_ids, embed_w})
                     .name("qwen3 embedding");
        for (int i = 0; i < static_cast<int>(layer_w.size()); ++i) {
            h = ctx.node("ggml_qwen3_layer", {B, L, hidden}, DType::F32, {h, attention_mask, layer_w[i]})
                    .name("qwen3 layer " + std::to_string(i));
            if (hidden_states && std::find(extract.begin(), extract.end(), i) != extract.end())
                hidden_states->push_back(h);
        }
    }
};

struct Transformer {
    Tensor in_w, out_w;
    std::vector<Tensor> block_w;
    int dim = 256;
    int token_dim = 64;

    // The analog of Flux2Transformer2DModel::forward. (Real: guidance =
    // nullopt — Klein is guidance-distilled; num_ref_tokens tells the model
    // where the reference tokens are — here the ids arrive pre-joined.)
    Tensor forward(Scope scope, Tensor hidden_states, Tensor encoder_hidden_states,
                   Tensor timestep, Tensor img_ids, Tensor txt_ids) {
        auto& ctx = scope.context();
        const auto B = hidden_states.shape()[0];
        const auto N = hidden_states.shape()[1];
        // Real: timestep = timestep * 1000.0f (the pipeline divides by 1000).
        timestep = timestep * 1000.0f;
        auto h = ctx.node("ggml_linear", {B, N, dim}, DType::F32, {hidden_states, in_w})
                     .name("transformer in_layer");
        for (int i = 0; i < static_cast<int>(block_w.size()); ++i)
            h = ctx.node("ggml_flux2_block", {B, N, dim}, DType::F32,
                         {h, encoder_hidden_states, timestep, img_ids, txt_ids, block_w[i]})
                    .name("transformer block " + std::to_string(i));
        return ctx.node("ggml_linear", {B, N, token_dim}, DType::F32, {h, out_w})
                   .name("transformer out_layer");
    }
};

// ----------------------------------------------------------------------------
// Host-side helpers (real: prepare_img_ids / prepare_txt_ids /
// prepare_ref_image_ids / the image preprocessing)
// ----------------------------------------------------------------------------

// Real: prepare_img_ids — the noise latents' ids: [batch, h*w, 4] =
// [0, y, x, 0] (GGML's RoPE wants 32-bit ints).
static std::vector<int32_t> make_img_ids(int batch, int64_t N, int packed_h, int packed_w) {
    std::vector<int32_t> ids(size_t(batch) * N * 4, 0);
    for (int b = 0; b < batch; ++b)
        for (int y = 0; y < packed_h; ++y)
            for (int x = 0; x < packed_w; ++x) {
                const auto off = ((size_t(b) * packed_h + y) * packed_w + x) * 4;
                ids[off + 1] = static_cast<int32_t>(y);
                ids[off + 2] = static_cast<int32_t>(x);
            }
    return ids;
}

static std::vector<int32_t> make_txt_ids(int batch, std::size_t L) {
    std::vector<int32_t> ids(size_t(batch) * L * 4, 0);
    for (int b = 0; b < batch; ++b)
        for (std::size_t l = 0; l < L; ++l)
            ids[(size_t(b) * L + l) * 4 + 3] = static_cast<int32_t>(l);
    return ids;
}

static std::vector<int32_t> make_ref_ids(int batch, std::size_t Nref, const std::vector<Image>& images,
                                         int vae_multiple) {
    std::vector<int32_t> ids(size_t(batch) * Nref * 4, 0);
    for (int b = 0; b < batch; ++b) {
        std::size_t token_offset = 0;
        for (std::size_t i = 0; i < images.size(); ++i) {
            const int h = images[i].height / vae_multiple;
            const int w = images[i].width / vae_multiple;
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) {
                    const auto off = (size_t(b) * Nref + token_offset) * 4;
                    ids[off] = 10 + static_cast<int>(i) * 10;
                    ids[off + 1] = static_cast<int32_t>(y);
                    ids[off + 2] = static_cast<int32_t>(x);
                    ++token_offset;
                }
        }
    }
    return ids;
}

// Real: image_to_tensor + VaeImageProcessor.preprocess — HWC uint8 -> CHW
// float, scaled to [-1, 1].
static std::vector<float> preprocess(const Image& img) {
    const auto w = img.width;
    const auto h = img.height;
    std::vector<float> data(3 * w * h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const auto src = (size_t(y) * w + x) * 3;
            const auto dst = size_t(y) * w + x;
            for (int c = 0; c < 3; ++c)
                data[c * w * h + dst] = 2.0f * (float(img.pixels[src + c]) / 255.0f) - 1.0f;
        }
    return data;
}

// ----------------------------------------------------------------------------
// Pure shape-conversion helpers (copied from Flux2KleinPipeline — they are
// Tensor ops, unchanged by the monad).
// ----------------------------------------------------------------------------

static Tensor patchify_latents(Tensor latents, int channels, int packed_h, int packed_w) {
    const auto B = latents.shape()[0];
    const auto C = channels;
    const auto N = packed_h * packed_w;
    auto t = latents.reshape({B * C, packed_h, 2, 2 * packed_w});
    t = t.permute({0, 2, 1, 3});
    t = t.reshape({B, 2 * C, packed_h, 2 * packed_w});
    t = t.permute({0, 1, 3, 2});
    t = t.reshape({B, 2 * C, N, 2});
    t = t.reshape({B, 4 * C, N});
    return t.reshape({B, 4 * C, packed_h, packed_w});
}

static Tensor pack_latents(Tensor latents) {
    const auto B = latents.shape()[0];
    const auto C4 = latents.shape()[1];
    const auto ph = latents.shape()[2];
    const auto pw = latents.shape()[3];
    const auto N = ph * pw;
    auto t = latents.reshape({B, C4, N});
    t = t.permute({0, 2, 1});
    return t.contiguous();
}

static Tensor unpack_latents(Tensor packed, int packed_h, int packed_w) {
    const auto B = packed.shape()[0];
    const auto C4 = packed.shape()[2];
    auto t = packed.permute({0, 2, 1}).contiguous();
    return t.reshape({B, C4, packed_h, packed_w});
}

static Tensor unpatchify_latents(Tensor patched, int channels, int packed_h, int packed_w) {
    const auto B = patched.shape()[0];
    const auto C = channels;
    const auto N = packed_h * packed_w;
    auto t = patched.reshape({B, 4 * C, N});
    t = t.reshape({B, 2 * C, 2, N});
    t = t.permute({0, 1, 3, 2}).contiguous();
    t = t.reshape({B, 2 * C, packed_h, 2 * packed_w});
    t = t.reshape({B * C, 2, packed_h, 2 * packed_w});
    t = t.permute({0, 2, 1, 3}).contiguous();
    t = t.reshape({B, C, 2 * packed_h, 2 * packed_w});
    return t;
}

// ============================================================================
// The pipeline (the analog of Flux2KleinPipeline)
// ============================================================================


struct Pipeline {
    Vae vae;
    TextEncoder text_encoder;
    Transformer transformer;

    // The monad rework of operator(): instead of building four graphs in
    // four contexts and driving four Computation objects (each allocating,
    // planning and executing one graph), this composes the whole generation
    // into one lazy description and returns the (unexecuted) computation of
    // the final image.
    Computation<Tensor> compute(
        Context& weights_ctx,
        const std::vector<int>& tokens,
        const std::vector<float>& mask,
        const std::vector<Image>& reference_images,
        const Options& options,
        const Schedule& schedule)
    {
        auto desc = std::make_shared<Description>();
        desc->weights = &weights_ctx;

        // Geometry (as in operator()).
        const auto vae_multiple = vae.scale_factor * 2;
        const auto target_height = (options.height / vae_multiple) * vae_multiple;
        const auto target_width = (options.width / vae_multiple) * vae_multiple;
        const auto packed_h = target_height / vae_multiple;  // = latent_height / 2
        const auto packed_w = target_width / vae_multiple;
        const auto N = packed_h * packed_w;  // the number of latent tokens
        const auto batch = options.num_images_per_prompt;
        const auto L = int64_t(tokens.size());  // the prompt length
        const auto token_dim = int64_t(vae.latent_channels) * 4;

        int64_t Nref = 0;
        for (auto& img : reference_images)
            Nref += (img.height / vae_multiple) * (img.width / vae_multiple);

        //
        // 1. VAE encode (img2img): reference images -> packed, normalized
        //    latents. A single-execution region; the output crosses into the
        //    denoise body (save).
        //
        std::optional<Computation<Tensor>> image_latents;
        if (!reference_images.empty()) {
            image_latents = save(region<Tensor>(desc, "vae encode",
                [&](Scope scope) -> Tensor {
                    auto& temp = scope.context();

                    std::vector<Tensor> packed_imgs;
                    for (std::size_t i = 0; i < reference_images.size(); ++i) {
                        const auto& img = reference_images[i];
                        // Real: image_to_tensor + VaeImageProcessor.preprocess
                        // (HWC uint8 -> CHW float, /255*2-1, resize_and_crop).
                        auto img_tensor = temp.create<float>(
                            {1, 3, img.height, img.width},
                            [&img](std::mt19937&) { return preprocess(img); });
                        img_tensor.name("reference image " + std::to_string(i));

                        auto mode = vae.encode(scope, img_tensor);
                        const auto ph = img.height / vae_multiple;
                        const auto pw = img.width / vae_multiple;
                        auto patched = patchify_latents(mode, vae.latent_channels, ph, pw);
                        auto bn_mean = vae.bn_mean.reshape({1, -1, 1, 1});
                        auto bn_std = sqrt(vae.bn_var.reshape({1, -1, 1, 1}) + vae.batch_norm_eps);
                        patched = (patched - bn_mean) / bn_std;
                        auto packed = pack_latents(patched);
                        packed.name("packed reference " + std::to_string(i));
                        packed_imgs.push_back(packed);
                    }
                    // Real: repeat_batch(packed, num_images_per_prompt) (a no-op
                    // for batch == 1).
                    auto out = Tensor::cat(packed_imgs, /*axis=*/1);
                    out.name("image_latents");
                    return out;
                }));
        }

        //
        // 2. Text encode: prompt -> embeddings. The stand-in stacks the
        //    hidden states of layers 0/2/3 (real: 9/18/27). The output
        //    crosses into the denoise body (save).
        //
        auto prompt_embeds = save(region<Tensor>(desc, "text encode",
            [&](Scope scope) -> Tensor {
                auto& temp = scope.context();

                // Real: tokenizer_.apply_chat_template + encode (host-side,
                // before the graph).
                auto input_ids = temp.create<int32_t>(
                    {batch, L},
                    [tokens, batch, L](std::mt19937&) {
                        std::vector<int32_t> ids(size_t(batch) * L);
                        for (int b = 0; b < batch; ++b)
                            std::copy(tokens.begin(), tokens.end(), ids.begin() + b * L);
                        return ids;
                    });
                input_ids.name("input_ids");

                auto attention_mask = temp.create<float>(
                    {batch, L},
                    [mask, batch, L](std::mt19937&) {
                        std::vector<float> m(size_t(batch) * L);
                        for (int b = 0; b < batch; ++b)
                            std::copy(mask.begin(), mask.end(), m.begin() + b * L);
                        return m;
                    });
                attention_mask.name("attention_mask");

                std::vector<Tensor> hidden_states;
                text_encoder.forward(scope, input_ids, attention_mask, &hidden_states);

                auto stacked = Tensor::stack(hidden_states, /*axis=*/1);
                auto out = stacked.permute({0, 2, 1, 3})
                               .reshape({batch, L, 3 * int64_t(text_encoder.hidden)});
                out.name("prompt_embeds");
                return out;
            }));

        //
        // 3. The denoising state: a state cell (the loop's "pure") — saved:
        //    the monad creates it in the state context, because it crosses
        //    the loop iterations and the decode region. Random noise, or the
        //    caller's initial latents.
        //
        std::function<std::vector<float>(std::mt19937&)> latents_provider;
        if (options.init_latents)
            latents_provider = [v = std::move(*options.init_latents)](std::mt19937&) { return v; };
        else
            latents_provider = [count = size_t(batch) * N * token_dim](std::mt19937& rng) {
                std::normal_distribution<float> normal;
                std::vector<float> noise(count);
                for (float& v : noise)
                    v = normal(rng);
                return noise;
            };
        auto latents = save<float>(desc, {batch, N, token_dim}, latents_provider, "latents");

        //
        // 4. Denoise — the generic monadic fold. The body is built once (one
        //    ggml graph, its own temporary context); the executor re-executes
        //    it per step: the per-step timestep/dt are re-bindable inputs
        //    (their providers iterate the schedule through the loop's iter
        //    clock — the same pattern as today's float* timestep capture, but
        //    owned by the loop), and the state is fed back after each step.
        //
        auto final_latents = fold(latents, schedule.size(),
            [&](Scope scope, Computation<Tensor> l, std::size_t& iter) -> Tensor {
                auto& temp = scope.context();

                auto img_ids = temp.create<int32_t>(
                    {batch, N, 4},
                    [batch, N, packed_h, packed_w](std::mt19937&) {
                        return make_img_ids(batch, N, packed_h, packed_w);
                    });
                img_ids.name("img_ids");

                auto txt_ids = temp.create<int32_t>(
                    {batch, L, 4},
                    [batch, L](std::mt19937&) {
                        return make_txt_ids(batch, L);
                    });
                txt_ids.name("txt_ids");

                Tensor latent_model_input = l.ref;
                Tensor latent_image_ids = img_ids;
                if (image_latents) {
                    auto ref_ids = temp.create<int32_t>(
                        {batch, Nref, 4},
                        [batch, Nref, &reference_images, vae_multiple](std::mt19937&) {
                            return make_ref_ids(batch, Nref, reference_images, vae_multiple);
                        });
                    ref_ids.name("ref_image_ids");

                    latent_model_input = Tensor::cat({l.ref, image_latents->ref}, /*axis=*/1);
                    latent_model_input.name("model input (latents + references)");
                    latent_image_ids = Tensor::cat({img_ids, ref_ids}, /*axis=*/1);
                    latent_image_ids.name("latent image ids");
                }

                // The per-step values: re-bindable inputs. The providers read
                // the loop's iter clock, so the same graph is re-executed with
                // the next step's values — no graph rebuild.
                auto timestep = temp.value<float>(
                    {batch},
                    [batch, &schedule, &iter](std::mt19937&) {
                        return std::vector<float>(batch, schedule[iter].timestep);
                    });
                timestep.name("timestep");

                auto dt = temp.value<float>(
                    {1},
                    [&schedule, &iter](std::mt19937&) {
                        return std::vector<float>{schedule[iter].dt};
                    });
                dt.name("dt");

                auto noise_pred = transformer.forward(
                    scope, latent_model_input, prompt_embeds.ref, timestep / 1000.0f,
                    latent_image_ids, txt_ids);

                if (image_latents) {
                    noise_pred = noise_pred.narrow(/*axis=*/1, 0, N);
                    noise_pred.name("noise_pred (slice to latents)");
                }

                // The scheduler's integrate: x + dt * model_output — the next
                // state (real: FlowMatchEulerDiscreteScheduler::integrate).
                auto next = l.ref + dt * noise_pred;
                next.name("next_latents");
                return next;
            },
            "denoise");

        //
        // 5. VAE decode: latents -> pixels. The output is the computation's
        //    result (saved into the state context).
        //
        auto image = save(region<Tensor>(desc, "vae decode",
            [&](Scope scope) -> Tensor {

                auto z_packed = unpack_latents(final_latents.ref, packed_h, packed_w);
                auto bn_mean = vae.bn_mean.reshape({1, -1, 1, 1});
                auto bn_std = sqrt(vae.bn_var.reshape({1, -1, 1, 1}) + vae.batch_norm_eps);
                z_packed = z_packed * bn_std + bn_mean;
                auto z = unpatchify_latents(z_packed, vae.latent_channels, packed_h, packed_w);
                auto out = vae.decode(scope, z);
                out.name("decoded image");
                return out;
            }));

        return image;
    }
};

// The analog of Flux2KleinPipeline::from_pretrained + GGUFLoaderVisitor: the
// tensors are created in the weights context (metadata, no buffer yet) and
// bound with providers streaming them from the GGUF file. The buffer comes
// from the one-shot allocator; the executor streams each binding once.
static void load_weights(Context& ctx, Pipeline& p) {
    auto gguf = [&](const Shape& shape, const std::string& name) -> Tensor {
        auto t = ctx.tensor(shape, DType::F32);
        t.name(name);
        const auto n = shape.numel();
        // Non-once, like the real GGUF bind — but the executor now writes it
        // exactly once (today: re-streamed by every Computation).
        ctx.bind_byte(t, [n](std::mt19937& rng) {
            std::vector<std::byte> bytes(size_t(n) * 4);
            std::uniform_int_distribution<int> u(0x30, 0x3F);
            for (auto& b : bytes)
                b = std::byte(u(rng));
            return bytes;
        });
        return t;
    };

    p.vae.enc0 = gguf({3, 32, 3, 3}, "vae.encoder.down0.weight");
    p.vae.enc1 = gguf({32, 16, 3, 3}, "vae.encoder.down1.weight");
    p.vae.dec0 = gguf({64, 32, 3, 3}, "vae.decoder.up0.weight");
    p.vae.dec1 = gguf({32, 16, 3, 3}, "vae.decoder.up1.weight");
    p.vae.dec2 = gguf({16, 3, 3, 3}, "vae.decoder.up2.weight");
    p.vae.bn_mean = gguf({p.vae.latent_channels}, "vae.bn.running_mean");
    p.vae.bn_var = gguf({p.vae.latent_channels}, "vae.bn.running_var");

    const int hidden = p.text_encoder.hidden;
    p.text_encoder.embed_w = gguf({4096, hidden}, "text_encoder.model.embed_tokens.weight");  // stand-in vocab
    for (int i = 0; i < 4; ++i)
        p.text_encoder.layer_w.push_back(
            gguf({hidden, hidden * 12}, "text_encoder.model.layers." + std::to_string(i) + ".fused"));

    const int dim = p.transformer.dim;
    p.transformer.in_w = gguf({p.transformer.token_dim, dim}, "transformer.net.in_layer.weight");
    p.transformer.out_w = gguf({dim, p.transformer.token_dim}, "transformer.net.out_layer.weight");
    for (int i = 0; i < 4; ++i)
        p.transformer.block_w.push_back(
            gguf({dim, dim * 12}, "transformer.net.layers." + std::to_string(i) + ".fused"));
}

// ============================================================================
// The run phase: the one-shot allocator, the scheduler, the executor
// ============================================================================

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

// The galloc model (ggml_gallocr, ggml-alloc.c): the high-water mark of the
// live set of the region's temporary context. A tensor is allocated when its
// producer runs (bound inputs: at the start) and freed after its last
// consumer (the graph's output: at the end — the executor reads it after the
// graph). Tensors that already have a buffer (weights, state) live in other
// contexts and are skipped by the galloc, as are views.
static std::size_t high_water(const Region& r) {
    const auto& tensors = r.temp->tensors();
    const auto& nodes = r.temp->nodes();
    if (nodes.empty())
        return 0;
    const int last = static_cast<int>(nodes.size()) - 1;
    std::vector<std::size_t> live_at(nodes.size(), 0);
    for (auto& m : tensors) {
        if (m.base >= 0)
            continue;  // a view: shares the base's buffer
        const int born = m.producer >= 0 ? m.producer : 0;  // bound inputs: at the start
        int death = last;  // the graph's output (and consumer-less tensors): until the end
        if (!m.consumers.empty())
            death = *std::max_element(m.consumers.begin(), m.consumers.end());
        for (int p = born; p <= death; ++p)
            live_at[p] += m.size_bytes;
    }
    return *std::max_element(live_at.begin(), live_at.end());
}

// The simplified allocator (the analog of src/ggml/Allocator):
//
// Today's allocator is incremental — use() per context, allocate() per
// Computation (i.e. per graph). In the monad the whole computation is known
// before execution (all the module forwards have been called), so the
// allocation is a single non-incremental pass over the long-lived contexts:
//
//   - the weights context (provided from outside, pinned)
//   - the state context (created on demand, lives for the whole generation)
//
// The temporary (per-graph) contexts are not allocated here: the scheduler
// allocates them per graph (they are discarded after the graph executes).
struct Allocator {
    void allocate(const std::vector<Context*>& contexts) {
        for (auto* ctx : contexts) {
            // The model of ggml_backend_alloc_ctx_tensors_from_buft
            // (ggml-alloc.c): a plain aligned bump over the whole context —
            // every tensor gets a slot (views and already-allocated tensors
            // are skipped), no liveness: the context lives for the whole
            // generation.
            std::size_t offset = 0;
            for (auto& m : ctx->tensors()) {
                if (m.base >= 0 || m.allocated)
                    continue;
                m.offset = (offset + 63) / 64 * 64;
                m.allocated = true;
                offset = m.offset + m.size_bytes;
            }
            ctx->buffer.resize(offset);
        }
    }
};

// The scheduler stand-in (the model of what ggml_backend_sched does to a
// graph):
//
//   alloc:   the galloc (ggml_gallocr) plans the graph's temporary tensors
//            with liveness; the compute buffer is sized to the high-water
//            mark of the live set. It is persistent and reused across graphs
//            (one buffer for the whole generation).
//   compute: walks the nodes in build order (the stand-in for
//            ggml_backend_sched_graph_compute).
struct Scheduler {
    std::size_t compute_buffer = 0;

    void alloc(Region& r) {
        r.high_water = high_water(r);
        compute_buffer = std::max(compute_buffer, r.high_water);
        std::cout << "     alloc: high-water " << format_bytes(r.high_water)
                  << "  (compute buffer: " << format_bytes(compute_buffer) << ")\n";
    }

    void compute(const Region& r, bool verbose) {
        const auto& nodes = r.temp->nodes();
        if (verbose) {
            for (std::size_t i = 0; i < nodes.size(); ++i) {
                const auto& n = nodes[i];
                std::cout << "        " << std::setw(2) << i << ": " << n.op << " '"
                          << r.temp->meta(n.output).name << "'";
                for (auto& in : n.inputs)
                    std::cout << " < " << in.name();
                std::cout << "\n";
            }
        } else {
            std::cout << "        compute: " << nodes.size() << " nodes (as before)\n";
        }
    }
};

// The executor — the run phase. It replaces the per-stage Computation
// objects: instead of four Computation objects (each: allocate + plan +
// execute one graph), one executor: one allocation pass, one execution pass
// over the regions.
class Executor {
public:
    Executor(const Computation<Tensor>& result, Allocator& allocator, Scheduler& scheduler,
             uint64_t seed = 0x5EED)
        : result_(result.ref), desc_(result.desc), scheduler_(scheduler), rng_(seed) {
        // The one-shot allocation point: all the module forwards have been
        // called — all the low-level ggml tensor chains exist. Allocate the
        // long-lived contexts in a single non-incremental pass.
        std::vector<Context*> long_lived;
        if (desc_->weights)
            long_lived.push_back(desc_->weights);
        if (desc_->state && desc_->state->has_tensors())
            long_lived.push_back(desc_->state.get());
        allocator.allocate(long_lived);
    }

    void dump() const;
    void run();
    std::vector<float> read_result() const;
    const Tensor& result() const { return result_; }

private:
    void write_bindings(Context* ctx, bool once_only);
    void copy(const Tensor& src, const Tensor& dst, const char* kind);

    Tensor result_;
    std::shared_ptr<Description> desc_;
    Scheduler& scheduler_;
    std::mt19937 rng_;
    std::size_t copy_count_ = 0;
};

void Executor::write_bindings(Context* ctx, bool once_only) {
    for (auto& [id, b] : ctx->bindings()) {
        if (b.unbound || b.once != once_only)
            continue;
        Tensor t{ctx, id};
        const auto& m = ctx->meta(id);
        auto bytes = b.provider(rng_);
        assert(bytes.size() == m.size_bytes);
        if (ctx->has_buffer())
            std::memcpy(ctx->buffer.data() + m.offset, bytes.data(), bytes.size());
        std::cout << "     write '" << m.name << "' " << m.shape.to_string() << " ["
                  << format_bytes(bytes.size()) << "] -> "
                  << (ctx->has_buffer() ? std::string(ctx->name()) + " buffer" : "compute buffer");
        // Print the value for small float tensors (the loop's timestep/dt).
        if (m.dtype == DType::F32 && m.shape.numel() > 0 && m.shape.numel() <= 8) {
            const auto* f = reinterpret_cast<const float*>(bytes.data());
            std::cout << "  = [";
            for (int64_t i = 0; i < m.shape.numel(); ++i) {
                if (i)
                    std::cout << ", ";
                char b[32];
                std::snprintf(b, sizeof b, "%.3f", f[i]);
                std::cout << b;
            }
            std::cout << "]";
        }
        std::cout << "\n";
    }
}

void Executor::copy(const Tensor& src, const Tensor& dst, const char* kind) {
    // The analog of ggml_backend_tensor_copy (the project's Context::copy):
    // a tensor in a temporary context is copied into the state context.
    // (The stand-in "computes" the copy: it fills the destination's slot.)
    const auto& d = dst.context().meta(dst.id);
    if (dst.context().has_buffer()) {
        auto* p = dst.context().buffer.data() + d.offset;
        for (std::size_t i = 0; i < d.size_bytes; ++i)
            p[i] = std::byte(static_cast<std::uint8_t>(0xA0 + ((i * 7 + copy_count_) & 0x1F)));
    }
    std::cout << "     copy '" << src.name() << "' (" << src.context().name() << ") -> '"
              << d.name << "' (" << dst.context().name() << ")  [" << kind << "]\n";
    ++copy_count_;
}

void Executor::run() {
    std::cout << "\n=== RUN ===\n";
    std::cout << "one pass over the regions (build order == execution order)\n\n";

    // The weights: the GGUF bindings are streamed once into the weights
    // buffer. (Today: re-streamed by every Computation.)
    if (desc_->weights) {
        std::cout << "  [weights] " << desc_->weights->tensors().size() << " tensors\n";
        write_bindings(desc_->weights, /*once_only=*/false);
    }

    // The state's one-shot bindings (the initial latents).
    if (desc_->state) {
        std::cout << "  [state]\n";
        write_bindings(desc_->state.get(), /*once_only=*/true);
    }
    std::cout << "\n";

    for (std::size_t i = 0; i < desc_->regions.size(); ++i) {
        auto& r = *desc_->regions[i];
        std::cout << "  -- region " << i << "/" << desc_->regions.size() - 1 << ": " << r.name;
        if (r.loop)
            std::cout << "  (loop x" << r.count << ")";
        std::cout << "\n";

        scheduler_.alloc(r);
        write_bindings(r.temp.get(), /*once_only=*/true);

        if (r.loop) {
            for (r.iter = 0; r.iter < r.count; ++r.iter) {
                std::cout << "     iter " << r.iter + 1 << "/" << r.count << ":\n";
                write_bindings(r.temp.get(), /*once_only=*/false);
                scheduler_.compute(r, r.iter == 0);
                // The feedback: the next state is written back into the
                // state cell (on every iteration, including the last — the
                // cell must hold the final state; today's loop never does
                // this, which is why its result is broken).
                copy(r.feedback->first, r.feedback->second, "feedback");
            }
        } else {
            scheduler_.compute(r, true);
        }

        for (auto& [s, d] : r.saves)
            copy(s, d, "boundary");

        // The temporary context is disposable: the graph has been computed,
        // the state is across the boundary. (Real: free the ggml context;
        // the compute buffer is kept by the scheduler.)
        std::cout << "     dispose temporary context '" << r.temp->name() << "'\n\n";
    }
}

void Executor::dump() const {
    const auto& d = *desc_;
    std::cout << "\n=== PLAN ===\n";

    std::cout << "the description (built by the monad — nothing allocated or executed yet):\n";
    for (std::size_t i = 0; i < d.regions.size(); ++i) {
        const auto& r = *d.regions[i];
        std::cout << "  " << i << ": " << std::left << std::setw(14) << r.name
                  << (r.loop ? std::string("[loop x") + std::to_string(r.count) + "]" : "[once]")
                  << std::right << "  " << r.temp->nodes().size() << " nodes in the temporary context";
        if (r.loop)
            std::cout << "\n                            feedback '" << r.feedback->first.name()
                      << "' -> '" << r.feedback->second.name() << "' after each iteration";
        std::cout << "\n";
    }

    if (d.state) {
        std::cout << "\n  the state context (created on demand — data that crosses graph boundaries):\n";
        for (auto& m : d.state->tensors())
            std::cout << "     " << std::left << std::setw(16) << m.name << std::right
                      << std::setw(22) << m.shape.to_string() << "  " << format_bytes(m.size_bytes) << "\n";
    }

    std::cout << "\none-shot allocation (all the module forwards have been called — all the ggml tensor chains exist):\n";

    std::size_t weights_total = 0;
    if (d.weights) {
        std::cout << "\n  the weights buffer (provided context, pinned — as today's USAGE_WEIGHTS):\n";
        for (auto& m : d.weights->tensors()) {
            weights_total += m.size_bytes;
            std::cout << "     " << std::left << std::setw(34) << m.name << std::right
                      << std::setw(14) << m.shape.to_string()
                      << "  @ " << std::setw(9) << format_bytes(m.offset)
                      << "  " << std::setw(9) << format_bytes(m.size_bytes) << "\n";
        }
        std::cout << "     total: " << format_bytes(weights_total) << "\n";
    }

    std::size_t state_total = 0;
    if (d.state) {
        std::cout << "\n  the state buffer (the state context — lives for the whole generation):\n";
        for (auto& m : d.state->tensors()) {
            state_total += m.size_bytes;
            std::cout << "     " << std::left << std::setw(16) << m.name << std::right
                      << std::setw(22) << m.shape.to_string()
                      << "  @ " << std::setw(9) << format_bytes(m.offset)
                      << "  " << std::setw(9) << format_bytes(m.size_bytes) << "\n";
        }
        std::cout << "     total: " << format_bytes(state_total) << "\n";
    }

    std::size_t max_hw = 0;
    std::size_t today_contexts = 0;
    std::cout << "\n  the compute buffer (the scheduler — per-graph liveness, reused across graphs):\n";
    for (auto& rp : d.regions) {
        const auto& r = *rp;
        const auto hw = high_water(r);
        max_hw = std::max(max_hw, hw);
        std::size_t sum = 0;
        for (auto& m : r.temp->tensors())
            if (m.base < 0)
                sum += m.size_bytes;  // views share the base's buffer
        today_contexts += sum;
        std::cout << "     " << std::left << std::setw(14) << r.name << std::right
                  << "  high-water " << std::setw(9) << format_bytes(hw);
        if (r.loop)
            std::cout << "   (x" << r.count << " — the buffer is reused)";
        std::cout << "\n";
    }
    std::cout << "     total: " << format_bytes(max_hw) << "\n";

    const auto peak = weights_total + state_total + max_hw;
    const auto today = weights_total + today_contexts;
    std::cout << "\n  peak memory (monad) = weights + state + compute = "
              << format_bytes(peak) << "\n";
    std::cout << "  peak memory (today) = weights + the sum of all the graph context buffers\n";
    std::cout << "  (no liveness, all the graphs coexist) = " << format_bytes(today) << "\n";
}

std::vector<float> Executor::read_result() const {
    const auto& ctx = result_.context();
    const auto& m = ctx.meta(result_.id);
    auto data = ctx.read<float>(result_);
    std::cout << "\n  read '" << m.name << "' " << m.shape.to_string() << " f32 ["
              << format_bytes(m.size_bytes) << "] from the '" << ctx.name() << "' buffer\n";
    return data;
}

// ============================================================================
// main
// ============================================================================

static std::vector<Image> make_reference_images(std::size_t n, int h, int w) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> u(0, 255);
    std::vector<Image> images;
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<std::byte> pixels(size_t(h) * w * 3);
        for (auto& p : pixels)
            p = std::byte(u(rng));
        images.push_back(Image{w, h, std::move(pixels)});
    }
    return images;
}

static Schedule make_schedule(int n) {
    // The stand-in of operator(): sigmas[i] = 1 - i/n (before shifting) —
    // the real one goes through FlowMatchEulerDiscreteScheduler::schedule.
    std::vector<float> sigmas(n + 1), timesteps(n);
    for (int i = 0; i < n; ++i) {
        sigmas[i] = 1.0f - float(i) / float(n);
        timesteps[i] = sigmas[i];
    }
    sigmas[n] = 0.0f;
    return Schedule(std::move(timesteps), std::move(sigmas));
}

int main() {
    // The weights context: provided from outside (from_pretrained loads the
    // modules into it). Pinned: allocated once, never freed.
    Context weights_ctx("weights");
    Pipeline pipeline;
    load_weights(weights_ctx, pipeline);

    // The host-side inputs (as in operator(): the tokenizer + the image
    // preprocessing run before the graph).
    const std::vector<int> tokens(48, 226);  // stand-in for the tokenizer output
    const std::vector<float> mask(48, 1.0f);
    const auto reference_images = make_reference_images(/*n=*/2, /*h=*/896, /*w=*/896);
    Options options;  // 1024x1024, 8 steps
    const auto schedule = make_schedule(options.num_inference_steps);

    // 1. Build — the monadic description of the whole generation.
    //    No allocation, no execution.
    auto image = pipeline.compute(weights_ctx, tokens, mask, reference_images, options, schedule);

    // 2. Plan + allocate — one-shot, now that all the module forwards have
    //    been called (all the low-level ggml tensor chains exist).
    Allocator allocator;
    Scheduler scheduler;
    Executor executable(image, allocator, scheduler);
    executable.dump();

    // 3. Run — one pass over the regions.
    executable.run();

    auto pixels = executable.read_result();
    // Real: latents_to_images(...) -> std::vector<Image>, then saved.
    std::cout << "  -> " << pixels.size() / 3 << " pixels (stand-in for latents_to_images / Image::save)\n";
    return 0;
}
