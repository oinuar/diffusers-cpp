// Tensor-split (meta device) integration test.
//
// Builds the "meta" virtual device over the CPU device twice. The meta backend
// initializes two independent CPU backends internally, which act as the two
// "GPUs". Runs a mul_mat chain that exercises the whole tensor-split machinery
// without any GPU:
//
//   a = W1 * x      W1 SPLIT axis1 (out rows, 12/12), x MIRRORED   -> a SPLIT axis0, zero comms
//   y = a * a       SPLIT axis0 x SPLIT axis0                      -> y PARTIAL  -> butterfly allreduce
//   z = W2 * y      W2 SPLIT axis1 (out rows, 6/6), y async MIRRORED -> z SPLIT axis0
//
// Notes on the graph shape (ggml 0.19.0 constraints, verified in ggml-backend-meta.cpp):
// - graph inputs are always assigned to the last (CPU) backend by the scheduler and
//   copied to the meta device as MIRRORED leaves, so x cannot be SPLIT; the PARTIAL
//   state is produced by the second mul_mat with the same SPLIT tensor on both sides
//   (y = a*a), which is exactly the "SPLIT axis0 x SPLIT axis0 -> PARTIAL" case.
// - the butterfly allreduce runs at the subgraph boundary after every PARTIAL node
//   except the last graph node, and a final graph node in PARTIAL state cannot be
//   read back (meta tensor_get has no PARTIAL case), so the graph must end in a
//   SPLIT/MIRRORED node that consumes the allreduced y.
// - if the allreduce did not happen, z would be W2_j * (a_j^T * a_j) per device and
//   the numeric check against the host reference below would fail, so a PASS
//   verifies scatter, split states, split-input copy, butterfly and gather readout.
//
// Run:          ctest --test-dir build -R meta-device
// Diagnostics:  GGML_META_DEBUG=1        (per-node split states)
//               GGML_SCHED_DEBUG=1       (node -> backend assignment)

#include "ggml/Backend.hpp"
#include "ggml/Computation.hpp"
#include "ggml/Context.hpp"
#include "ggml/Graph.hpp"
#include "ggml/MetaDevice.hpp"
#include "ggml/Runtime.hpp"
#include "ggml/Scheduler.hpp"
#include "ggml/SplitState.hpp"
#include "ggml/Tensor.hpp"

#include <ggml.h>
#include <ggml-backend.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static void require(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "meta-device FAIL: " << what << std::endl;
        std::exit(1);
    }
}

int main() {
    ggml_time_init();
    ggml_log_set([](ggml_log_level, const char* text, void*) { std::cerr << text; }, nullptr);
    ggml_backend_load_all();

    // --- 1. meta device over two CPU backends -------------------------------
    // The CPU backend registers a single device; passing it twice makes the
    // meta backend initialize two independent CPU backends (one per "GPU").
    const auto cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    require(cpu_dev != nullptr, "no CPU device found");

    const int n_devices = 2;
    MetaDevice meta_dev(std::vector<ggml_backend_dev_t>(n_devices, cpu_dev));
    require(*meta_dev != nullptr, "ggml_backend_meta_device failed");

    Backend meta_backend(*meta_dev);
    require(*meta_backend != nullptr, "failed to init meta backend");
    Backend cpu_backend(GGML_BACKEND_DEVICE_TYPE_CPU); // plain CPU stays last (ggml requirement)

    std::cout << "meta backend: " << ggml_backend_name(*meta_backend) << std::endl;
    std::cout << "cpu  backend: " << ggml_backend_name(*cpu_backend) << std::endl;

    // --- 2. problem dimensions -----------------------------------------------
    const int64_t K = 32; // input features (contraction dim of first matmul)
    const int64_t N = 24; // mid features, W1 rows split N/n_devices per device
    const int64_t P = 16; // batch
    const int64_t Q = 12; // output features, W2 rows split Q/n_devices per device

    // --- 3. runtime ------------------------------------------------------------
    Scheduler scheduler(std::vector<ggml_backend_t>{*meta_backend, *cpu_backend});
    Context context(scheduler.capacity());
    Runtime runtime(scheduler, context);

    // --- 4. weights in meta buffers with explicit split specs -------------------
    // Weight values are generated on the host so the double-precision reference
    // below uses exactly the same data; the providers return those values.
    std::vector<float> w1_data(static_cast<size_t>(N) * K), w2_data(static_cast<size_t>(Q) * P);
    {
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& v : w1_data)
            v = dist(rng);
        for (auto& v : w2_data)
            v = dist(rng);
    }

    // Each weight lives in its own meta buffer (usage WEIGHTS) and splits its
    // output rows (ggml axis 1) evenly across the devices. The split state must
    // be registered before the tensor is allocated, so the buffer prepares the
    // tensor through the hook.
    auto split_rows = [&](ggml_tensor* t) {
        meta_dev.split(t, SplitState::split(/*ggml axis=*/1, t->ne[1], n_devices));
    };

    auto W1 = runtime.create_weight<float>(
        Tensor::Shape{N, K},
        [w1_data](std::mt19937&) { return w1_data; },
        meta_dev.buffer_type(),
        split_rows);

    auto W2 = runtime.create_weight<float>(
        Tensor::Shape{Q, P},
        [w2_data](std::mt19937&) { return w2_data; },
        meta_dev.buffer_type(),
        split_rows);

    // Verify the scatter: reading the SPLIT weights back through the meta
    // buffer must reproduce the host data exactly.
    auto verify_scatter = [&](Tensor& t, const std::vector<float>& data) {
        auto readback = runtime.value<float>(t);
        require(readback == data, "scatter verify: SPLIT weight readback mismatch");
        std::cout << "scatter verify: SPLIT axis1 weight roundtrip ok (" << ggml_get_name(*t) << ")" << std::endl;
    };
    verify_scatter(W1, w1_data);
    verify_scatter(W2, w2_data);

    // --- 5. input + graph --------------------------------------------------------
    // Input values are generated on the host so the double-precision reference
    // below uses exactly the same data.
    std::vector<float> x_data(static_cast<size_t>(P) * K);
    {
        std::mt19937 rng(4321);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& v : x_data)
            v = dist(rng);
    }

    Tensor x = runtime.create<float>(
        Tensor::Shape{P, K},
        [&](std::mt19937&) { return x_data; });

    ggml_context* ctx = *runtime.context();
    // a = W1 x :  W1 ne {K,N} SPLIT1, x ne {K,P} MIRRORED -> a ne {N,P} SPLIT0
    Tensor a = Tensor(ctx, ggml_mul_mat(ctx, *W1, *x), Tensor::Shape{P, N});
    // y = a a :  SPLIT0 x SPLIT0 -> y ne {P,P} PARTIAL -> butterfly allreduce
    Tensor y = Tensor(ctx, ggml_mul_mat(ctx, *a, *a), Tensor::Shape{P, P});
    // z = W2 y :  W2 ne {P,Q} SPLIT1, y (async) MIRRORED -> z ne {Q,P} SPLIT0
    Tensor z = Tensor(ctx, ggml_mul_mat(ctx, *W2, *y), Tensor::Shape{P, Q});

    (void)y;
    Graph graph(runtime, {z});
    Computation computation(graph);

    auto z_meta = runtime.value<float>(z);

    // --- 6. double-precision host reference ---------------------------------------
    // ggml_mul_mat(W, v) with W ne {K, N}: res[i, j] = sum_k W.data[i*K + k] * v.data[j*K + k]
    std::vector<double> A(static_cast<size_t>(N) * P), Y(static_cast<size_t>(P) * P), Z(static_cast<size_t>(Q) * P);
    for (int64_t i = 0; i < N; i++) {
        for (int64_t j = 0; j < P; j++) {
            double s = 0.0;
            for (int64_t k = 0; k < K; k++)
                s += static_cast<double>(w1_data[i * K + k]) * static_cast<double>(x_data[j * K + k]);

            A[i * P + j] = s;
        }
    }
    for (int64_t i = 0; i < P; i++) {
        for (int64_t j = 0; j < P; j++) {
            double s = 0.0;
            for (int64_t k = 0; k < N; k++)
                s += A[k * P + i] * A[k * P + j];

            Y[i * P + j] = s;
        }
    }
    for (int64_t i = 0; i < Q; i++) {
        for (int64_t j = 0; j < P; j++) {
            double s = 0.0;
            for (int64_t k = 0; k < P; k++)
                s += static_cast<double>(w2_data[i * P + k]) * Y[k * P + j];

            Z[i * P + j] = s;
        }
    }

    // z_meta is PyTorch [P, Q]: z_meta[j*Q + i] == Z[i*P + j]
    double max_diff = 0.0;
    for (int64_t i = 0; i < Q; i++) {
        for (int64_t j = 0; j < P; j++)
            max_diff = std::max(max_diff, std::abs(static_cast<double>(z_meta[j * Q + i]) - Z[i * P + j]));
    }

    const double tolerance = 1e-4;
    std::cout << "max |z_meta - z_ref| = " << max_diff << " (tolerance " << tolerance << ")" << std::endl;
    require(max_diff < tolerance, "numeric mismatch vs host reference");

    std::cout << "PASS: meta(CPU,CPU) mul_mat graph with SPLIT weights, PARTIAL allreduce and SPLIT readout" << std::endl;
    return 0;
}
