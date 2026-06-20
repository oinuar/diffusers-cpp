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

class MatMulComputation : public GGMLComputation {
public:
    virtual std::pair<Dependencies, Tensor> build(GGMLContext& ctx) {
        // create tensors
        auto ta = Tensor::empty<2>(*ctx, GGML_TYPE_F32, {cols_A, rows_A});
        auto tb = Tensor::empty<2>(*ctx, GGML_TYPE_F32, {cols_B, rows_B});

        // result = a*b^T
        auto result = Tensor(*ctx, ggml_mul_mat(*ctx, *ta, *tb));

        return {{ta, tb}, result};
    }

    virtual void compute(GGMLGraph& graph) {
        {
            GGMLScope scope(graph);
            auto& inputs = scope.inputs();

            // load data from cpu memory to backend buffer
            ggml_backend_tensor_set(*inputs[0], matrix_A, 0, ggml_nbytes(*inputs[0]));
            ggml_backend_tensor_set(*inputs[1], matrix_B, 0, ggml_nbytes(*inputs[1]));
        }

        // in this case, the output tensor is the last one in the graph
        auto output = graph.node(-1);

        // create a array to print result
        std::vector<float> out_data(ggml_nelements(output));

        // bring the data from the backend memory
        ggml_backend_tensor_get(output, out_data.data(), 0, ggml_nbytes(output));

        printf("mul mat (%d x %d) (transposed result):\n[", (int) output->ne[0], (int) output->ne[1]);
        for (int j = 0; j < output->ne[1] /* rows */; j++) {
            if (j > 0) {
                printf("\n");
            }

            for (int i = 0; i < output->ne[0] /* cols */; i++) {
                printf(" %.2f", out_data[j * output->ne[0] + i]);
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

    MatMulComputation matmul;

    auto graph = scheduler.plan(matmul);
    matmul.compute(graph);

    return 0;
}
