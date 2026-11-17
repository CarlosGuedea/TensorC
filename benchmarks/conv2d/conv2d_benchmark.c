/*
 * Benchmark de Conv2D. Lee la entrada y el kernel desde archivos binarios
 * float32 (row-major) para garantizar que la comparación contra NumPy use
 * exactamente los mismos datos y dimensiones.
 *
 * Uso:
 *   conv2d_benchmark input.bin kernel.bin N C H W K KH KW SH SW PH PW reps
 *
 * Salida (parseable):
 *   FORWARD_MS_MIN <ms>
 *   FORWARD_MS_MEDIAN <ms>
 *   BACKWARD_MS_MIN <ms>
 *   BACKWARD_MS_MEDIAN <ms>
 */
#include <tensorc/tensorc.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static float *read_floats(const char *path, size_t count) {
    FILE *file = fopen(path, "rb");
    if (!file) return NULL;
    float *data = (float *)malloc(count * sizeof(float));
    if (!data || fread(data, sizeof(float), count, file) != count) {
        free(data);
        data = NULL;
    }
    fclose(file);
    return data;
}

static int compare_doubles(const void *a, const void *b) {
    const double x = *(const double *)a;
    const double y = *(const double *)b;
    return (x > y) - (x < y);
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

/* Devuelve la mediana y el mínimo en milisegundos. */
static void summarize(const double *samples, size_t count,
                      double *median_out, double *min_out) {
    double *sorted = (double *)malloc(count * sizeof(double));
    if (!sorted) {
        *median_out = samples[0];
        *min_out = samples[0];
        return;
    }
    memcpy(sorted, samples, count * sizeof(double));
    qsort(sorted, count, sizeof(double), compare_doubles);
    *median_out = sorted[count / 2];
    *min_out = sorted[0];
    free(sorted);
}

int main(int argc, char **argv) {
    if (argc != 16) {
        fprintf(stderr,
                "uso: %s input.bin kernel.bin N C H W K KH KW SH SW PH PW reps\n",
                argv[0]);
        return 2;
    }
    const size_t shape[4] = {(size_t)atoll(argv[3]), (size_t)atoll(argv[4]),
                             (size_t)atoll(argv[5]), (size_t)atoll(argv[6])};
    const size_t kernel_shape[4] = {(size_t)atoll(argv[7]), (size_t)atoll(argv[8]),
                                    (size_t)atoll(argv[9]), (size_t)atoll(argv[10])};
    const tc_conv2d_options options = {(size_t)atoll(argv[11]),
                                       (size_t)atoll(argv[12]),
                                       (size_t)atoll(argv[13]),
                                       (size_t)atoll(argv[14])};
    const size_t reps = (size_t)atoll(argv[15]);
    size_t input_count = 1, kernel_count = 1;
    for (int i = 0; i < 4; ++i) {
        input_count *= shape[i];
        kernel_count *= kernel_shape[i];
    }

    float *input_data = read_floats(argv[1], input_count);
    float *kernel_data = read_floats(argv[2], kernel_count);
    if (!input_data || !kernel_data) {
        fprintf(stderr, "no se pudieron leer los datos de entrada\n");
        free(input_data);
        free(kernel_data);
        return 2;
    }

    tc_tensor_nd *input = tc_tensor_nd_from_data(shape, 4, input_data, 1);
    tc_tensor_nd *kernel = tc_tensor_nd_from_data(kernel_shape, 4, kernel_data, 1);
    free(input_data);
    free(kernel_data);
    if (!input || !kernel) {
        fprintf(stderr, "no se pudieron crear los tensores\n");
        tc_tensor_nd_free(kernel);
        tc_tensor_nd_free(input);
        return 2;
    }

    double *forward_ms = (double *)malloc(reps * sizeof(double));
    double *backward_ms = (double *)malloc(reps * sizeof(double));
    if (!forward_ms || !backward_ms) {
        free(forward_ms);
        free(backward_ms);
        tc_tensor_nd_free(kernel);
        tc_tensor_nd_free(input);
        return 2;
    }

    /* Calentamiento: una pasada hacia adelante y hacia atrás. */
    tc_tensor_nd *warm = tc_tensor_nd_conv2d_ex(input, kernel, options);
    tc_tensor_nd *warm_loss = warm ? tc_tensor_nd_sum(warm) : NULL;
    if (warm_loss) tc_tensor_nd_backward(warm_loss);

    /* Volcar la salida para que el script NumPy verifique la exactitud. */
    FILE *dump = fopen("c_output.bin", "wb");
    if (dump) {
        fwrite(tc_tensor_nd_const_data(warm), sizeof(float),
               tc_tensor_nd_numel(warm), dump);
        fclose(dump);
    }
    tc_tensor_nd_free(warm_loss);
    tc_tensor_nd_free(warm);

    for (size_t r = 0; r < reps; ++r) {
        const double start = now_ms();
        tc_tensor_nd *output = tc_tensor_nd_conv2d_ex(input, kernel, options);
        forward_ms[r] = now_ms() - start;
        tc_tensor_nd_free(output);
    }

    /* Backward: el grafo se reutiliza; cada backward recalcula gradientes. */
    tc_tensor_nd *output = tc_tensor_nd_conv2d_ex(input, kernel, options);
    tc_tensor_nd *loss = tc_tensor_nd_sum(output);
    for (size_t r = 0; r < reps; ++r) {
        const double start = now_ms();
        if (!tc_tensor_nd_backward(loss)) {
            fprintf(stderr, "backward falló\n");
            break;
        }
        backward_ms[r] = now_ms() - start;
    }

    double forward_median, forward_min, backward_median, backward_min;
    summarize(forward_ms, reps, &forward_median, &forward_min);
    summarize(backward_ms, reps, &backward_median, &backward_min);
    printf("FORWARD_MS_MIN %.4f\n", forward_min);
    printf("FORWARD_MS_MEDIAN %.4f\n", forward_median);
    printf("BACKWARD_MS_MIN %.4f\n", backward_min);
    printf("BACKWARD_MS_MEDIAN %.4f\n", backward_median);

    free(forward_ms);
    free(backward_ms);
    tc_tensor_nd_free(loss);
    tc_tensor_nd_free(output);
    tc_tensor_nd_free(kernel);
    tc_tensor_nd_free(input);
    return 0;
}
