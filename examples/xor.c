#include "../minitf.h"

#include <stdio.h>
#include <time.h>

static double wall_time_seconds(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

int main(void) {
    mt_seed(42);

    /* Four XOR examples, one example per matrix row. */
    MTTensor x = mt_tensor_create(4, 2);
    MTTensor y = mt_tensor_create(4, 1);
    if (!x.data || !y.data) {
        fprintf(stderr, "No se pudo reservar memoria para los datos.\n");
        mt_tensor_free(&x); mt_tensor_free(&y); return 1;
    }
    const float x_values[] = {0,0, 0,1, 1,0, 1,1};
    const float y_values[] = {0,1,1,0};
    for (size_t i = 0; i < 8; ++i) x.data[i] = x_values[i];
    for (size_t i = 0; i < 4; ++i) y.data[i] = y_values[i];

    const size_t layer_sizes[] = {8, 1};
    const MTActivation activations[] = {MT_SIGMOID, MT_SIGMOID};
    MTModel model = mt_model_create(2, layer_sizes, activations, 2);
    if (!model.layers) {
        fprintf(stderr, "No se pudo crear el modelo.\n");
        mt_tensor_free(&x); mt_tensor_free(&y); return 1;
    }

    const size_t epochs = 12000;
    double start = wall_time_seconds();
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        float loss = mt_model_train_epoch(&model, &x, &y, 1.0f);
        if ((epoch + 1) % 2000 == 0)
            printf("Época %zu/%zu | MSE = %.6f\n", epoch + 1, epochs, loss);
    }
    double elapsed = wall_time_seconds() - start;
    printf("Tiempo de entrenamiento C: %.6f s\n", elapsed);

    puts("\nPredicciones XOR:");
    MTTensor input = mt_tensor_create(1, 2);
    for (size_t row = 0; row < x.rows; ++row) {
        input.data[0] = x.data[row * x.cols];
        input.data[1] = x.data[row * x.cols + 1];
        MTTensor output = mt_model_predict(&model, &input);
        printf("%.0f XOR %.0f = %.4f -> clase %d (real %.0f)\n",
               input.data[0], input.data[1], output.data[0],
               output.data[0] >= 0.5f, y.data[row]);
        mt_tensor_free(&output);
    }

    mt_tensor_free(&input);
    mt_model_free(&model);
    mt_tensor_free(&x);
    mt_tensor_free(&y);
    return 0;
}
