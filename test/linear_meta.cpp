// Linear tensor-split (meta device) integration test.
//
// Builds the "meta" virtual device over two CPU backends (as in meta_device.cpp)
// and runs a real nn::Linear module on it. Unlike meta_device.cpp, no split state
// is registered by hand: the Linear module marks its weight and bias to be
// sharded along their output-feature axis, and the CreateParametersVisitor
// materializes them as static weights in a meta buffer (usage WEIGHTS) with the
// split states registered at allocation time:
//
//   y = W x + b    W SPLIT axis1 (out rows, 4/2), b SPLIT axis0 (out rows, 4/2),
//                  x MIRRORED   -> y SPLIT axis0, zero communication
//
// Checks performed:
// - the weight and bias actually end up SPLIT on the meta device (is_split),
//   so a regression to mirrored weights is detected even though the numbers
//   would still be correct,
// - the SPLIT parameters read back through the meta buffer reproduce the
//   host data exactly,
// - the sharded ggml_mul_mat and split bias add match the double-precision
//   host reference (the SPLIT output is gathered on readout).
//
// Run:          ctest --test-dir build -R linear-meta
// Diagnostics:  GGML_META_DEBUG=1        (per-node split states)
//               GGML_SCHED_DEBUG=1       (node -> backend assignment)

#include "ggml/Backend.hpp"
#include "ggml/Computation.hpp"
#include "ggml/Context.hpp"
#include "ggml/Graph.hpp"
#include "ggml/MetaDevice.hpp"
#include "ggml/Runtime.hpp"
#include "ggml/Scheduler.hpp"
#include "ggml/Tensor.hpp"
#include "nn/Linear.hpp"
#include "nn/Parameter.hpp"
#include "nn/RethrowVisitor.hpp"
#include "nn/Visitor.hpp"
#include "utils/TestCLI.hpp"

#include <ggml.h>
#include <ggml-backend.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static void require(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "linear-meta FAIL: " << what << std::endl;
        std::exit(1);
    }
}

// Serializes a row-major float tensor as the nested list literal understood by
// the test CLIs' TensorParser. %.9g round-trips float32 values exactly.
static std::string to_list_literal(const std::vector<float>& data, const Tensor::Shape& shape) {
    size_t offset = 0;

    std::function<std::string(size_t)> build = [&](size_t dim) -> std::string {
        std::string s = "[";

        for (int64_t i = 0; i < shape[dim]; ++i) {
            if (dim + 1 < shape.rank())
                s += build(dim + 1);
            else {
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%.9g", data[offset++]);
                s += buf;
            }

            if (i + 1 < shape[dim])
                s += ", ";
        }

        return s + "]";
    };

    return build(0);
}

class CollectParameters : public Visitor {
public:
    virtual void visit(Parameter& parameter, std::vector<std::string> path) {
        if (path.size() == 1 && path[0] == "weight")
            weight = &parameter;

        if (path.size() == 1 && path[0] == "bias")
            bias = &parameter;
    }

    Parameter* weight = nullptr;
    Parameter* bias = nullptr;
};

int main() {
    ggml_time_init();
    ggml_log_set([](ggml_log_level, const char* text, void*) { std::cerr << text; }, nullptr);
    ggml_backend_load_all();

    // --- 1. meta device over two CPU backends -------------------------------
    const auto cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    require(cpu_dev != nullptr, "no CPU device found");

    const int n_devices = 2;
    MetaDevice meta_dev(std::vector<ggml_backend_dev_t>(n_devices, cpu_dev));
    require(*meta_dev != nullptr, "ggml_backend_meta_device failed");

    Backend meta_backend(*meta_dev);
    require(*meta_backend != nullptr, "failed to init meta backend");
    Backend cpu_backend(GGML_BACKEND_DEVICE_TYPE_CPU); // plain CPU stays last (ggml requirement)

    std::cout << "meta backend: " << ggml_backend_name(*meta_backend) << std::endl;

    // --- 2. problem dimensions -----------------------------------------------
    const int64_t in_features = 8;
    const int64_t out_features = 4; // split out_features / n_devices rows per device
    const int64_t batch = 3;

    // --- 3. runtime ------------------------------------------------------------
    Scheduler scheduler(std::vector<ggml_backend_t>{*meta_backend, *cpu_backend});
    Context context(scheduler.capacity());
    Runtime runtime(scheduler, context, std::random_device{}(), &meta_dev);

    // Weight/bias values are generated on the host so the double-precision
    // reference below uses exactly the same data.
    std::vector<float> w_data(static_cast<size_t>(out_features) * in_features);
    std::vector<float> b_data(static_cast<size_t>(out_features));
    std::vector<float> x_data(static_cast<size_t>(batch) * in_features);
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& v : w_data)
            v = dist(rng);
        for (auto& v : b_data)
            v = dist(rng);
        for (auto& v : x_data)
            v = dist(rng);
    }

    // --- 4. Linear module, parameters through the real CLI visitor -------------
    // The parameter values travel as CLI arguments exactly like in the test
    // CLIs, so the CreateParametersVisitor meta path (static meta-buffer weight
    // + split state registration) is the one under test.
    std::vector<std::string> argv = {
        "Linear",
        "--in_features", std::to_string(in_features),
        "--out_features", std::to_string(out_features),
        "--param-weight", to_list_literal(w_data, Tensor::Shape{out_features, in_features}),
        "--param-bias", to_list_literal(b_data, Tensor::Shape{out_features}),
    };
    std::vector<char*> args;
    for (auto& s : argv)
        args.push_back(s.data());

    Linear model(in_features, out_features, /*bias=*/true);

    ArgumentParser args_parser(args.size(), args.data());
    TestCLI::CreateParametersVisitor create_parameters(runtime, args_parser);
    RethrowVisitor visitor(create_parameters);
    model.accept(visitor);

    try {
        visitor.rethrow();
    } catch (const std::exception& e) {
        std::cerr << "linear-meta FAIL: " << e.what() << std::endl;
        std::exit(1);
    }

    CollectParameters collect;
    model.accept(collect);
    require(collect.weight != nullptr && collect.bias != nullptr, "Linear weight/bias not found");

    // Sharding proof: the split states must be registered on the actual
    // weight/bias tensors (a silent fallback to mirrored weights would still
    // produce correct numbers, so this is asserted explicitly).
    auto weight_tensor = collect.weight->operator*();
    auto bias_tensor = collect.bias->operator*();
    require(meta_dev.is_split(*weight_tensor), "Linear weight is not split across the meta devices");
    require(meta_dev.is_split(*bias_tensor), "Linear bias is not split across the meta devices");
    std::cout << "split verify: weight and bias SPLIT across " << n_devices << " devices" << std::endl;

    // Scatter verify: reading the SPLIT parameters back through the meta
    // buffer must reproduce the host data exactly.
    auto verify_scatter = [&](const Tensor& t, const std::vector<float>& data) {
        auto readback = runtime.value<float>(t);
        require(readback == data, "scatter verify: SPLIT parameter readback mismatch");
    };
    verify_scatter(weight_tensor, w_data);
    verify_scatter(bias_tensor, b_data);
    std::cout << "scatter verify: SPLIT weight/bias roundtrip ok" << std::endl;

    // --- 5. forward + graph -----------------------------------------------------
    Tensor x = runtime.create<float>(
        Tensor::Shape{batch, in_features},
        [&](std::mt19937&) { return x_data; });

    Tensor output = model.forward(runtime, x);

    Graph graph(runtime, {output});
    Computation computation(graph);

    auto y_meta = runtime.value<float>(output);

    // --- 6. double-precision host reference ---------------------------------------
    // y[j, i] = sum_k w_data[i*in + k] * x_data[j*in + k] + b_data[i]
    std::vector<double> ref(static_cast<size_t>(out_features) * batch);
    for (int64_t j = 0; j < batch; ++j) {
        for (int64_t i = 0; i < out_features; ++i) {
            double s = static_cast<double>(b_data[i]);
            for (int64_t k = 0; k < in_features; ++k)
                s += static_cast<double>(w_data[static_cast<size_t>(i) * in_features + k]) * static_cast<double>(x_data[static_cast<size_t>(j) * in_features + k]);

            ref[static_cast<size_t>(i) * batch + j] = s;
        }
    }

    // y_meta is PyTorch [batch, out_features]: y_meta[j*out + i] == ref[i*batch + j]
    double max_diff = 0.0;
    for (int64_t i = 0; i < out_features; ++i) {
        for (int64_t j = 0; j < batch; ++j)
            max_diff = std::max(max_diff, std::abs(static_cast<double>(y_meta[static_cast<size_t>(j) * out_features + i]) - ref[static_cast<size_t>(i) * batch + j]));
    }

    const double tolerance = 1e-4;
    std::cout << "max |y_meta - y_ref| = " << max_diff << " (tolerance " << tolerance << ")" << std::endl;
    require(max_diff < tolerance, "numeric mismatch vs host reference");

    std::cout << "PASS: Linear on meta(CPU,CPU) with SPLIT weight/bias, sharded mul_mat and SPLIT readout" << std::endl;
    return 0;
}
