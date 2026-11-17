#include <tensorc/tensorc.h>

#include <math.h>
#include <stdio.h>

static int close_enough(float actual, float expected) {
    return fabsf(actual - expected) < 1e-5f;
}

static float conv2d_loss_value(tc_tensor_nd *input, tc_tensor_nd *kernel) {
    tc_tensor_nd *output = tc_tensor_nd_conv2d(input, kernel);
    tc_tensor_nd *loss = tc_tensor_nd_sum(output);
    const float value = loss ? tc_tensor_nd_const_data(loss)[0] : NAN;
    tc_tensor_nd_free(loss);
    tc_tensor_nd_free(output);
    return value;
}

static int finite_difference_check(tc_tensor_nd *input, tc_tensor_nd *kernel) {
    const float epsilon = 1e-3f;
    float *input_data = tc_tensor_nd_data(input);
    float *kernel_data = tc_tensor_nd_data(kernel);
    const float *input_gradient = tc_tensor_nd_grad(input);
    const float *kernel_gradient = tc_tensor_nd_grad(kernel);

    for (size_t i = 0; i < tc_tensor_nd_numel(input); ++i) {
        const float original = input_data[i];
        input_data[i] = original + epsilon;
        const float plus = conv2d_loss_value(input, kernel);
        input_data[i] = original - epsilon;
        const float minus = conv2d_loss_value(input, kernel);
        input_data[i] = original;
        const float numeric = (plus - minus) / (2.0f * epsilon);
        if (fabsf(input_gradient[i] - numeric) > 2e-3f) return 0;
    }

    for (size_t i = 0; i < tc_tensor_nd_numel(kernel); ++i) {
        const float original = kernel_data[i];
        kernel_data[i] = original + epsilon;
        const float plus = conv2d_loss_value(input, kernel);
        kernel_data[i] = original - epsilon;
        const float minus = conv2d_loss_value(input, kernel);
        kernel_data[i] = original;
        const float numeric = (plus - minus) / (2.0f * epsilon);
        if (fabsf(kernel_gradient[i] - numeric) > 2e-3f) return 0;
    }
    return 1;
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

    ok = ok && finite_difference_check(input, kernel);

    const size_t incompatible_shape[] = {1, 2, 2, 2};
    const float incompatible_values[] = {1, 1, 1, 1, 1, 1, 1, 1};
    tc_tensor_nd *incompatible = tc_tensor_nd_from_data(
        incompatible_shape, 4, incompatible_values, 0);
    tc_tensor_nd *invalid_output = tc_tensor_nd_conv2d(input, incompatible);
    ok = ok && incompatible && !invalid_output;

    printf("conv2d: %s\n", ok ? "ok" : "error");
    tc_tensor_nd_free(invalid_output);
    tc_tensor_nd_free(incompatible);
    tc_tensor_nd_free(loss);
    tc_tensor_nd_free(output);
    tc_tensor_nd_free(kernel);
    tc_tensor_nd_free(input);
    return ok ? 0 : 1;
}
