#include <tensorc/tensorc.h>

#include <math.h>
#include <stdio.h>

static int close_enough(float actual, float expected) {
    return fabsf(actual - expected) < 1e-6f;
}

int main(void) {
    const char *path = "tensorc_model_roundtrip.tcm";
    const size_t sizes[] = {3, 2};
    const tc_activation activations[] = {TC_RELU, TC_SIGMOID};
    tc_model original = tc_model_create(2, sizes, activations, 2);
    if (!original.layers) return 1;

    for (size_t l = 0; l < original.layer_count; ++l) {
        for (size_t i = 0; i < original.layers[l].input_size * original.layers[l].output_size; ++i)
            original.layers[l].weights[i] = (float)(i + 1) * 0.1f + (float)l;
        for (size_t j = 0; j < original.layers[l].output_size; ++j)
            original.layers[l].biases[j] = (float)(j + 1) * -0.05f;
    }

    tc_tensor input = tc_tensor_create(1, 2);
    input.data[0] = 0.25f;
    input.data[1] = -0.75f;
    tc_tensor expected = tc_model_predict(&original, &input);
    int ok = input.data && expected.data && tc_model_save(&original, path);

    tc_model loaded = tc_model_load(path);
    tc_tensor actual = {0};
    if (loaded.layers) actual = tc_model_predict(&loaded, &input);
    ok = ok && loaded.layers && loaded.input_size == original.input_size &&
         loaded.layer_count == original.layer_count && actual.data &&
         actual.rows == expected.rows && actual.cols == expected.cols;
    for (size_t l = 0; ok && l < original.layer_count; ++l) {
        ok = loaded.layers[l].input_size == original.layers[l].input_size &&
             loaded.layers[l].output_size == original.layers[l].output_size &&
             loaded.layers[l].activation == original.layers[l].activation;
        for (size_t i = 0; ok && i < original.layers[l].input_size * original.layers[l].output_size; ++i)
            ok = close_enough(loaded.layers[l].weights[i], original.layers[l].weights[i]);
        for (size_t j = 0; ok && j < original.layers[l].output_size; ++j)
            ok = close_enough(loaded.layers[l].biases[j], original.layers[l].biases[j]);
    }
    for (size_t i = 0; ok && i < expected.rows * expected.cols; ++i)
        ok = close_enough(actual.data[i], expected.data[i]);

    printf("serialization: %s\n", ok ? "ok" : "error");
    tc_tensor_free(&actual);
    tc_tensor_free(&expected);
    tc_tensor_free(&input);
    tc_model_free(&loaded);
    tc_model_free(&original);
    remove(path);
    return ok ? 0 : 1;
}
