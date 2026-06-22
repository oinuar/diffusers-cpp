#include "GGMLCompute.hpp"
#include "GGMLComputation.hpp"
#include "GGMLBackend.hpp"
#include "GGMLScheduler.hpp"
#include <iostream>

// initialize data of matrices to perform matrix multiplication
const int rows_A = 4, cols_A = 2;

float matrix_A[rows_A * cols_A] = {
    2, 8,
    5, 1,
    4, 2,
    8, 6
};

const int rows_B = 3, cols_B = 2;
/* Transpose([
    10, 9, 5,
    5, 9, 4
]) 2 rows, 3 cols */
float matrix_B[rows_B * cols_B] = {
    10, 5,
    9, 9,
    5, 4
};

class MatMul : public GGMLCompute {
public:
    virtual std::pair<Parameters, Tensor> build(GGMLContext& ctx) {
        // create tensors
        auto ta = Tensor::empty<2>(*ctx, GGML_TYPE_F32, {cols_A, rows_A});
        auto tb = Tensor::empty<2>(*ctx, GGML_TYPE_F32, {cols_B, rows_B});

        // result = a*b^T
        auto result = Tensor(*ctx, ggml_mul_mat(*ctx, *ta, *tb));

        return {{{"a", ta}, {"b", tb}}, result};
    }

    virtual void compute(GGMLGraph& graph) {
        GGMLComputation computation(graph);

        // Do the computation with matrices
        computation({
            {"a", matrix_A},
            {"b", matrix_B}
        });

        // Extract the result. In this case, the output tensor is the last one in the graph.
        auto [shape, data] = computation.get<float>(-1);

        printf("mul mat (%d x %d) (transposed result):\n[", (int)shape[0], (int)shape[1]);
        for (int j = 0; j < shape[1] /* rows */; j++) {
            if (j > 0) {
                printf("\n");
            }

            for (int i = 0; i < shape[0] /* cols */; i++) {
                printf(" %.2f", data[j * shape[0] + i]);
            }
        }
        printf(" ]\n");
    }
};

int main() {
    ggml_time_init();
    ggml_log_set([](ggml_log_level, const char* text, void*) { std::cerr << text; }, nullptr);

    ggml_backend_load_all();

    GGMLBackend cpu(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGMLScheduler scheduler({*cpu});

    MatMul matmul;

    auto graph = scheduler.plan(matmul);
    matmul.compute(graph);

    return 0;
}
