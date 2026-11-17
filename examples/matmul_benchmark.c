#include <tensorc/tensorc.h>

#include <stdio.h>
#include <time.h>

#define MATRIX_SIZE 512
#define REPEATS 8

int main(void) {
    tc_tensor a = tc_tensor_create(MATRIX_SIZE, MATRIX_SIZE);
    tc_tensor b = tc_tensor_create(MATRIX_SIZE, MATRIX_SIZE);
    if (!a.data || !b.data) {
        fprintf(stderr, "TensorC: no se pudo reservar memoria para matmul.\n");
        tc_tensor_free(&a);
        tc_tensor_free(&b);
        return 1;
    }
    for (size_t i = 0; i < a.rows * a.cols; ++i) {
        a.data[i] = (float)((int)(i % 17) - 8) * 0.01f;
        b.data[i] = (float)((int)(i % 13) - 6) * 0.02f;
    }

    tc_tensor result = {0, 0, NULL};
    const clock_t start = clock();
    float checksum = 0.0f;
    for (size_t repeat = 0; repeat < REPEATS; ++repeat) {
        tc_tensor_free(&result);
        result = tc_tensor_matmul(&a, &b);
        if (!result.data) {
            tc_tensor_free(&a);
            tc_tensor_free(&b);
            return 1;
        }
        checksum += result.data[0] + result.data[result.rows * result.cols - 1];
    }
    const clock_t end = clock();
    const double elapsed_ms = 1000.0 * (double)(end - start) / (double)CLOCKS_PER_SEC;
    printf("matrix=%dx%d repeats=%d\n", MATRIX_SIZE, MATRIX_SIZE, REPEATS);
    printf("elapsed_ms=%.3f\n", elapsed_ms);
    printf("checksum=%.6f\n", checksum);

    tc_tensor_free(&result);
    tc_tensor_free(&a);
    tc_tensor_free(&b);
    return 0;
}
