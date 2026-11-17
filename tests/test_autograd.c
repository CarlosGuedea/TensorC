#include <tensorc/tensorc.h>

#include <math.h>
#include <stdio.h>

static int close_enough(float actual, float expected) {
    return fabsf(actual - expected) < 1e-5f;
}

int main(void) {
    const size_t shape[] = {2, 2};
    const float a_data[] = {1.0f, -2.0f, 3.0f, 4.0f};
    const float b_data[] = {2.0f, 5.0f, -1.0f, 2.0f};
    tc_tensor_nd *a = tc_tensor_nd_from_data(shape, 2, a_data, 1);
    tc_tensor_nd *b = tc_tensor_nd_from_data(shape, 2, b_data, 1);
    tc_tensor_nd *product = tc_tensor_nd_mul(a, b);
    tc_tensor_nd *loss = tc_tensor_nd_sum(product);
    if (!a || !b || !product || !loss || !tc_tensor_nd_backward(loss)) return 1;

    const float expected_a[] = {2.0f, 5.0f, -1.0f, 2.0f};
    const float expected_b[] = {1.0f, -2.0f, 3.0f, 4.0f};
    const float *grad_a = tc_tensor_nd_grad(a);
    const float *grad_b = tc_tensor_nd_grad(b);
    for (size_t i = 0; i < 4; ++i) {
        if (!close_enough(grad_a[i], expected_a[i]) ||
            !close_enough(grad_b[i], expected_b[i])) {
            fprintf(stderr, "gradiente incorrecto en índice %zu\n", i);
            return 1;
        }
    }

    tc_tensor_nd_free(loss);
    tc_tensor_nd_free(product);
    tc_tensor_nd_free(b);
    tc_tensor_nd_free(a);
    puts("autograd: ok");
    return 0;
}
