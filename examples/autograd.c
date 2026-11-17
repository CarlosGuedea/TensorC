#include <tensorc/tensorc.h>

#include <stdio.h>

int main(void) {
    const size_t matrix_shape[] = {2, 2};
    const float matrix_data[] = {1.0f, -2.0f, 3.0f, 4.0f};
    const float target_data[] = {2.0f, 1.0f, 4.0f, 8.0f};

    tc_tensor_nd *weights = tc_tensor_nd_from_data(matrix_shape, 2, matrix_data, 1);
    tc_tensor_nd *target = tc_tensor_nd_from_data(matrix_shape, 2, target_data, 0);
    tc_tensor_nd *prediction = tc_tensor_nd_relu(weights);
    tc_tensor_nd *loss = tc_tensor_nd_mse_loss(prediction, target);
    if (!weights || !target || !prediction || !loss || !tc_tensor_nd_backward(loss)) {
        fprintf(stderr, "TensorC: no se pudo ejecutar autodiferenciación.\n");
        tc_tensor_nd_free(loss);
        tc_tensor_nd_free(prediction);
        tc_tensor_nd_free(target);
        tc_tensor_nd_free(weights);
        return 1;
    }

    tc_tensor_nd_print(prediction, "relu(weights)");
    printf("MSE = %.5f\n", tc_tensor_nd_const_data(loss)[0]);
    printf("grad(weights) = [");
    const float *gradient = tc_tensor_nd_grad(weights);
    for (size_t i = 0; i < tc_tensor_nd_numel(weights); ++i)
        printf("%s%.5f", i ? ", " : "", gradient[i]);
    printf("]\n");

    tc_tensor_nd_free(loss);
    tc_tensor_nd_free(prediction);
    tc_tensor_nd_free(target);
    tc_tensor_nd_free(weights);
    return 0;
}
