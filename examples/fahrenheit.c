#include <tensorc/tensorc.h>

#include <math.h>
#include <stdio.h>
#include <time.h>

#define SAMPLE_COUNT 2049
#define EPOCHS 500
#define BATCH_SIZE 64
#define LEARNING_RATE 0.20f
#define FAHRENHEIT_MIN (-40.0f)
#define FAHRENHEIT_MAX 212.0f
#define CELSIUS_MIN (-40.0f)
#define CELSIUS_MAX 100.0f

static float normalize_fahrenheit(float fahrenheit) {
    return (fahrenheit - FAHRENHEIT_MIN) /
           (FAHRENHEIT_MAX - FAHRENHEIT_MIN);
}

static float denormalize_celsius(float normalized) {
    return normalized * (CELSIUS_MAX - CELSIUS_MIN) + CELSIUS_MIN;
}

int main(void) {
    tc_tensor x = tc_tensor_create(SAMPLE_COUNT, 1);
    tc_tensor y = tc_tensor_create(SAMPLE_COUNT, 1);
    if (!x.data || !y.data) {
        fprintf(stderr, "TensorC: no se pudo reservar el dataset.\n");
        tc_tensor_free(&x);
        tc_tensor_free(&y);
        return 1;
    }

    for (size_t i = 0; i < SAMPLE_COUNT; ++i) {
        const float fahrenheit = FAHRENHEIT_MIN +
                                 (FAHRENHEIT_MAX - FAHRENHEIT_MIN) *
                                     (float)i / (float)(SAMPLE_COUNT - 1);
        const float celsius = (fahrenheit - 32.0f) * (5.0f / 9.0f);
        x.data[i] = normalize_fahrenheit(fahrenheit);
        y.data[i] = (celsius - CELSIUS_MIN) / (CELSIUS_MAX - CELSIUS_MIN);
    }

    const size_t layer_sizes[] = {1};
    const tc_activation activations[] = {TC_LINEAR};
    tc_model model = tc_model_create(1, layer_sizes, activations, 1);
    if (!model.layers) {
        fprintf(stderr, "TensorC: no se pudo crear el modelo.\n");
        tc_tensor_free(&x);
        tc_tensor_free(&y);
        return 1;
    }
    /* Fixed initialization makes the C/Python benchmark reproducible. */
    model.layers[0].weights[0] = 0.0f;
    model.layers[0].biases[0] = 0.0f;

    tc_optimizer optimizer = tc_optimizer_sgd(LEARNING_RATE);
    const clock_t start = clock();
    float loss = 0.0f;
    for (size_t epoch = 0; epoch < EPOCHS; ++epoch)
        loss = tc_model_train_epoch_opt(&model, &x, &y, &optimizer, BATCH_SIZE);
    const clock_t end = clock();

    tc_tensor sample = tc_tensor_create(1, 1);
    sample.data[0] = normalize_fahrenheit(100.0f);
    tc_tensor prediction = tc_model_predict(&model, &sample);
    const float celsius_prediction = denormalize_celsius(prediction.data[0]);
    const double elapsed_ms = 1000.0 * (double)(end - start) / (double)CLOCKS_PER_SEC;

    printf("language=C\n");
    printf("samples=%d epochs=%d batch_size=%d\n", SAMPLE_COUNT, EPOCHS, BATCH_SIZE);
    printf("elapsed_ms=%.3f\n", elapsed_ms);
    printf("final_mse=%.8f\n", loss);
    printf("100F_to_C=%.5f\n", celsius_prediction);

    tc_tensor_free(&prediction);
    tc_tensor_free(&sample);
    tc_optimizer_free(&optimizer);
    tc_model_free(&model);
    tc_tensor_free(&x);
    tc_tensor_free(&y);
    return 0;
}
