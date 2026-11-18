/* Benchmark del API de grafo (tc_tensor_nd_matmul): es la ruta que tiene
 * backend CUDA. Warmup incluido para no medir la inicialización de CUDA. */
#include <tensorc/ndtensor.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 512
#define WARMUP 3
#define REPEATS 20

static double now_ms(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(void) {
    const size_t shape[] = {N, N};
    float *values = (float *)malloc((size_t)N * N * sizeof(float));
    if (!values) return 1;
    for (size_t i = 0; i < (size_t)N * N; ++i)
        values[i] = (float)((int)(i % 17) - 8) * 0.01f;
    tc_tensor_nd *a = tc_tensor_nd_from_data(shape, 2, values, 0);
    for (size_t i = 0; i < (size_t)N * N; ++i)
        values[i] = (float)((int)(i % 13) - 6) * 0.02f;
    tc_tensor_nd *b = tc_tensor_nd_from_data(shape, 2, values, 0);
    free(values);
    if (!a || !b) return 1;

    printf("cuda_available=%d\n", tc_cuda_available());
    float checksum = 0.0f;
    for (int i = 0; i < WARMUP; ++i) {
        tc_tensor_nd *c = tc_tensor_nd_matmul(a, b);
        checksum += tc_tensor_nd_const_data(c)[0];
        tc_tensor_nd_free(c);
    }
    double best = 1e30;
    double total = 0.0;
    for (int i = 0; i < REPEATS; ++i) {
        const double t0 = now_ms();
        tc_tensor_nd *c = tc_tensor_nd_matmul(a, b);
        const double t1 = now_ms();
        checksum += tc_tensor_nd_const_data(c)[0];
        tc_tensor_nd_free(c);
        const double ms = t1 - t0;
        total += ms;
        if (ms < best) best = ms;
    }
    printf("matrix=%dx%d repeats=%d warmup=%d\n", N, N, REPEATS, WARMUP);
    printf("mean_ms=%.3f min_ms=%.3f\n", total / REPEATS, best);
    printf("checksum=%.6f\n", checksum);
    tc_tensor_nd_free(a);
    tc_tensor_nd_free(b);
    return 0;
}
