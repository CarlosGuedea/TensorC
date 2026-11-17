#include <tensorc/tensorc.h>

#include <stdio.h>

int main(void) {
    tc_seed(42);
    tc_tensor x = tc_tensor_create(4, 2);
    tc_tensor y = tc_tensor_create(4, 1);
    if (!x.data || !y.data) return 1;
    const float x_data[] = {0, 0, 0, 1, 1, 0, 1, 1};
    const float y_data[] = {0, 1, 1, 0};
    for (size_t i = 0; i < 8; ++i) x.data[i] = x_data[i];
    for (size_t i = 0; i < 4; ++i) y.data[i] = y_data[i];

    const size_t sizes[] = {8, 1};
    const tc_activation activations[] = {TC_SIGMOID, TC_SIGMOID};
    tc_model model = tc_model_create(2, sizes, activations, 2);
    tc_optimizer optimizer = tc_optimizer_adam(&model, 0.05f);
    if (!model.layers || !optimizer.m_weights) return 1;

    for (size_t epoch = 0; epoch < 3000; ++epoch)
        tc_model_train_epoch_opt(&model, &x, &y, &optimizer, 4);

    tc_tensor sample = tc_tensor_create(1, 2);
    for (size_t row = 0; row < x.rows; ++row) {
        sample.data[0] = x.data[row * x.cols];
        sample.data[1] = x.data[row * x.cols + 1];
        tc_tensor prediction = tc_model_predict(&model, &sample);
        printf("%.0f XOR %.0f = %.4f\n", sample.data[0], sample.data[1], prediction.data[0]);
        tc_tensor_free(&prediction);
    }

    tc_tensor_free(&sample);
    tc_optimizer_free(&optimizer);
    tc_model_free(&model);
    tc_tensor_free(&x);
    tc_tensor_free(&y);
    return 0;
}
