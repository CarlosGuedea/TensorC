#include <tensorc/tensorc.h>

#include <math.h>
#include <stdio.h>

static int close_enough(float actual, float expected) {
    return fabsf(actual - expected) < 1e-5f;
}

int main(void) {
    const size_t input_shape[] = {1, 1, 3, 3};
    const size_t kernel_shape[] = {1, 1, 2, 2};
    const float input_values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    const float kernel_values[] = {1, 0, 0, 1};
    const float expected_output[] = {6, 8, 12, 14};
    const float expected_input_grad[] = {1, 1, 0, 1, 2, 1, 0, 1, 1};
    const float expected_kernel_grad[] = {12, 16, 24, 28};

    tc_tensor_nd *input = tc_tensor_nd_from_data(input_shape, 4, input_values, 1);
    tc_tensor_nd *kernel = tc_tensor_nd_from_data(kernel_shape, 4, kernel_values, 1);
    tc_tensor_nd *output = tc_tensor_nd_conv2d(input, kernel);
    tc_tensor_nd *loss = tc_tensor_nd_sum(output);
    int ok = input && kernel && output && loss && tc_tensor_nd_numel(output) == 4 &&
             tc_tensor_nd_backward(loss);
    const float *output_data = tc_tensor_nd_const_data(output);
    const float *input_grad = tc_tensor_nd_grad(input);
    const float *kernel_grad = tc_tensor_nd_grad(kernel);
    for (size_t i = 0; ok && i < 4; ++i)
        ok = close_enough(output_data[i], expected_output[i]);
    for (size_t i = 0; ok && i < 9; ++i)
        ok = close_enough(input_grad[i], expected_input_grad[i]);
    for (size_t i = 0; ok && i < 4; ++i)
        ok = close_enough(kernel_grad[i], expected_kernel_grad[i]);

    printf("conv2d: %s\n", ok ? "ok" : "error");
    tc_tensor_nd_free(loss);
    tc_tensor_nd_free(output);
    tc_tensor_nd_free(kernel);
    tc_tensor_nd_free(input);
    return ok ? 0 : 1;
}
