#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include <string>
#include <cstdio>
#include <cstring>

void demo() {
    printf("demo running...\n");
    const int rows_A = 4, cols_A = 2;
    float matrix_A[rows_A * cols_A] = {
        2, 8,
        5, 1,
        4, 2,
        8, 6
    };

    const int rows_B = 3, cols_B = 2;
    float matrix_B[rows_B * cols_B] = {
        10, 5,
        9, 9,
        5, 4
    };

    size_t ctx_size = 0;
    ctx_size += rows_A * cols_A * ggml_type_size(GGML_TYPE_F32);
    ctx_size += rows_B * cols_B * ggml_type_size(GGML_TYPE_F32);
    ctx_size += rows_A * rows_B * ggml_type_size(GGML_TYPE_F32);
    ctx_size += 3 * ggml_tensor_overhead();    // metadata for 3 tensors
    ctx_size += ggml_tensor_overhead();        // compute graph
    ctx_size += 1024;    // some overhead (exact calculation omitted for simplicity)

    struct ggml_init_params params = {
        .mem_size   = ctx_size,
        .mem_buffer = NULL,
        .no_alloc   = false
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * tensor_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_A, rows_A);
    struct ggml_tensor * tensor_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_B, rows_B);
    memcpy(tensor_a->data, matrix_A, ggml_nbytes(tensor_a));
    memcpy(tensor_b->data, matrix_B, ggml_nbytes(tensor_b));

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    struct ggml_tensor * result = ggml_mul_mat(ctx, tensor_a, tensor_b);  // result = a*b^T
    // Mark the "result" tensor to be computed
    ggml_build_forward_expand(gf, result);

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend == NULL) {
        fprintf(stderr, "%s: failed to initialize CPU backend\n", __func__);
        ggml_free(ctx);
        return;
    }

    enum ggml_status status = ggml_backend_graph_compute(backend, gf);
    
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: computation failed with status %d\n", __func__, status);
    }

    float * result_data = (float *) result->data;
    printf("mul mat (%d x %d) (transposed result):\n[", (int) result->ne[0], (int) result->ne[1]);
    for (int j = 0; j < result->ne[1]/* rows */; j++) {
        if (j > 0) {
            printf("\n");
        }

        for (int i = 0; i < result->ne[0]/* cols */; i++) {
            printf(" %.2f", result_data[j * result->ne[0] + i]);
        }
    }
    printf(" ]\n");

    ggml_backend_free(backend);
    ggml_free(ctx);
}

int main() {
    demo();
    return 0;
}