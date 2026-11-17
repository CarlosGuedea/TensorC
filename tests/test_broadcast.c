#include <tensorc/tensorc.h>

#include <math.h>
#include <stdio.h>

static int close_enough(float actual, float expected) {
    return fabsf(actual - expected) < 1e-5f;
}

static int check_values(const tc_tensor_nd *tensor, const float *expected,
                        size_t count) {
    if (!tensor || tc_tensor_nd_numel(tensor) != count) return 0;
    const float *data = tc_tensor_nd_const_data(tensor);
    for (size_t i = 0; i < count; ++i)
        if (!close_enough(data[i], expected[i])) return 0;
    return 1;
}

int main(void) {
    const size_t matrix_shape[] = {2, 3};
    const size_t vector_shape[] = {3};
    const float matrix_data[] = {1, 2, 3, 4, 5, 6};
    const float vector_data[] = {10, 20, 30};
    tc_tensor_nd *matrix = tc_tensor_nd_from_data(matrix_shape, 2, matrix_data, 1);
    tc_tensor_nd *vector = tc_tensor_nd_from_data(vector_shape, 1, vector_data, 1);
    if (!matrix || !vector) return 1;

    tc_tensor_nd *sum = tc_tensor_nd_add(matrix, vector);
    const float expected_sum[] = {11, 22, 33, 14, 25, 36};
    tc_tensor_nd *sum_loss = tc_tensor_nd_sum(sum);
    int ok = check_values(sum, expected_sum, 6) && sum_loss &&
             tc_tensor_nd_backward(sum_loss);
    const float *matrix_grad = tc_tensor_nd_grad(matrix);
    const float *vector_grad = tc_tensor_nd_grad(vector);
    for (size_t i = 0; ok && i < 6; ++i) ok = close_enough(matrix_grad[i], 1.0f);
    for (size_t i = 0; ok && i < 3; ++i) ok = close_enough(vector_grad[i], 2.0f);
    tc_tensor_nd_free(sum_loss);
    tc_tensor_nd_free(sum);

    tc_tensor_nd *product = tc_tensor_nd_mul(matrix, vector);
    const float expected_product[] = {10, 40, 90, 40, 100, 180};
    tc_tensor_nd *product_loss = tc_tensor_nd_sum(product);
    ok = ok && check_values(product, expected_product, 6) && product_loss &&
         tc_tensor_nd_backward(product_loss);
    matrix_grad = tc_tensor_nd_grad(matrix);
    vector_grad = tc_tensor_nd_grad(vector);
    const float expected_matrix_grad[] = {10, 20, 30, 10, 20, 30};
    const float expected_vector_grad[] = {5, 7, 9};
    for (size_t i = 0; ok && i < 6; ++i)
        ok = close_enough(matrix_grad[i], expected_matrix_grad[i]);
    for (size_t i = 0; ok && i < 3; ++i)
        ok = close_enough(vector_grad[i], expected_vector_grad[i]);

    printf("broadcast: %s\n", ok ? "ok" : "error");
    tc_tensor_nd_free(product_loss);
    tc_tensor_nd_free(product);
    tc_tensor_nd_free(vector);
    tc_tensor_nd_free(matrix);
    return ok ? 0 : 1;
}
