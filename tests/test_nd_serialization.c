#include <tensorc/tensorc.h>

#include <math.h>
#include <stdio.h>
#include <string.h>

static int close_enough(float actual, float expected) {
    return fabsf(actual - expected) < 1e-6f;
}

/* Roundtrip de un tensor único: forma, datos y requires_grad. */
static int test_single_roundtrip(void) {
    const char *path = "/tmp/tensorc_nd_single.tcb";
    const size_t shape[] = {2, 3, 4};
    const size_t numel = 2 * 3 * 4;
    float values[2 * 3 * 4];
    for (size_t i = 0; i < numel; ++i)
        values[i] = (float)((int)(i * 13 % 17) - 8) * 0.125f;
    tc_tensor_nd *original = tc_tensor_nd_from_data(shape, 3, values, 0);
    int ok = original && tc_tensor_nd_save(original, path);

    tc_tensor_nd *loaded = tc_tensor_nd_load(path, 1);
    ok = ok && loaded && tc_tensor_nd_ndim(loaded) == 3 &&
         tc_tensor_nd_shape(loaded)[0] == 2 &&
         tc_tensor_nd_shape(loaded)[1] == 3 &&
         tc_tensor_nd_shape(loaded)[2] == 4 &&
         tc_tensor_nd_requires_grad(loaded);
    const float *data = tc_tensor_nd_const_data(loaded);
    for (size_t i = 0; ok && i < numel; ++i)
        ok = close_enough(data[i], values[i]);

    tc_tensor_nd_free(loaded);
    tc_tensor_nd_free(original);
    remove(path);
    return ok;
}

/* Roundtrip de un checkpoint de varios tensores. */
static int test_checkpoint_roundtrip(void) {
    const char *path = "/tmp/tensorc_nd_checkpoint.tck";
    const size_t shape_a[] = {4, 4};
    const size_t shape_b[] = {1, 3, 2};
    const float values_a[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    const float values_b[] = {-0.5f, 0.25f, 1.5f, -2.0f, 0.0f, 3.25f};
    tc_tensor_nd *a = tc_tensor_nd_from_data(shape_a, 2, values_a, 0);
    tc_tensor_nd *b = tc_tensor_nd_from_data(shape_b, 3, values_b, 0);
    const tc_tensor_nd *tensors[] = {a, b};
    int ok = a && b && tc_save_checkpoint(tensors, 2, path);

    size_t count = 0;
    tc_tensor_nd **loaded = tc_load_checkpoint(path, &count, 0);
    ok = ok && loaded && count == 2;
    ok = ok && tc_tensor_nd_shape(loaded[0])[0] == 4 &&
         tc_tensor_nd_shape(loaded[0])[1] == 4 &&
         tc_tensor_nd_shape(loaded[1])[0] == 1 &&
         tc_tensor_nd_shape(loaded[1])[1] == 3 &&
         tc_tensor_nd_shape(loaded[1])[2] == 2;
    const float *data_a = tc_tensor_nd_const_data(loaded[0]);
    const float *data_b = tc_tensor_nd_const_data(loaded[1]);
    for (size_t i = 0; ok && i < 16; ++i)
        ok = close_enough(data_a[i], values_a[i]);
    for (size_t i = 0; ok && i < 6; ++i)
        ok = close_enough(data_b[i], values_b[i]);

    tc_free_checkpoint(loaded, count);
    tc_tensor_nd_free(b);
    tc_tensor_nd_free(a);
    remove(path);
    return ok;
}

/* Rechazos: archivo inexistente, mágico incorrecto y archivo truncado. */
static int test_rejections(void) {
    const char *path = "/tmp/tensorc_nd_corrupt.tcb";
    int ok = tc_tensor_nd_load("/tmp/tensorc_no_existe.tcb", 0) == NULL;

    const size_t shape[] = {2, 2};
    const float values[] = {1, 2, 3, 4};
    tc_tensor_nd *tensor = tc_tensor_nd_from_data(shape, 2, values, 0);
    ok = ok && tensor && tc_tensor_nd_save(tensor, path);

    /* Archivo con mágico inválido. */
    FILE *file = fopen(path, "wb");
    ok = ok && file && fwrite("XXXXXXXX", 1, 8, file) == 8;
    if (file) fclose(file);
    ok = ok && tc_tensor_nd_load(path, 0) == NULL;
    {
        size_t count = 42;
        tc_tensor_nd **checkpoint = tc_load_checkpoint(path, &count, 0);
        ok = ok && checkpoint == NULL && count == 0;
    }

    /* Archivo truncado: cabecera válida pero sin datos. */
    file = fopen(path, "wb");
    ok = ok && file;
    if (file) {
        const char magic[] = "TCND0001";
        const unsigned long long ndim = 2, rows = 2, cols = 2;
        ok = ok && fwrite(magic, 1, 8, file) == 8 &&
             fwrite(&ndim, sizeof(ndim), 1, file) == 1 &&
             fwrite(&rows, sizeof(rows), 1, file) == 1 &&
             fwrite(&cols, sizeof(cols), 1, file) == 1;
        fclose(file);
    }
    ok = ok && tc_tensor_nd_load(path, 0) == NULL;

    tc_tensor_nd_free(tensor);
    remove(path);
    return ok;
}

int main(void) {
    const int single_ok = test_single_roundtrip();
    const int checkpoint_ok = test_checkpoint_roundtrip();
    const int rejections_ok = test_rejections();
    printf("single roundtrip: %s\n", single_ok ? "ok" : "error");
    printf("checkpoint roundtrip: %s\n", checkpoint_ok ? "ok" : "error");
    printf("rejections: %s\n", rejections_ok ? "ok" : "error");
    return (single_ok && checkpoint_ok && rejections_ok) ? 0 : 1;
}
