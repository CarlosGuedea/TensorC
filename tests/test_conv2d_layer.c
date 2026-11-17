#include <tensorc/tensorc.h>

#include <math.h>
#include <stdio.h>

static int close_enough(float actual, float expected) {
    return fabsf(actual - expected) < 1e-5f;
}

static int gradient_close(float analytic, float numeric) {
    return fabsf(analytic - numeric) <= 2e-3f * (1.0f + fabsf(numeric));
}

/* Diferencias finitas centrales para loss(x) = sum(op(x)). */
static int finite_difference(const tc_tensor_nd *x, float (*loss_fn)(tc_tensor_nd *),
                             const float *analytic) {
    const float epsilon = 1e-2f;
    float *data = tc_tensor_nd_data((tc_tensor_nd *)x);
    for (size_t i = 0; i < tc_tensor_nd_numel(x); ++i) {
        const float original = data[i];
        data[i] = original + epsilon;
        const float plus = loss_fn((tc_tensor_nd *)x);
        data[i] = original - epsilon;
        const float minus = loss_fn((tc_tensor_nd *)x);
        data[i] = original;
        const float numeric = (plus - minus) / (2.0f * epsilon);
        if (!gradient_close(analytic[i], numeric)) return 0;
    }
    return 1;
}

static float reshape_loss(tc_tensor_nd *x) {
    const size_t flat[] = {tc_tensor_nd_numel(x), 1};
    tc_tensor_nd *out = tc_tensor_nd_reshape_to(x, flat, 2);
    tc_tensor_nd *loss = tc_tensor_nd_sum(out);
    const float value = loss ? tc_tensor_nd_const_data(loss)[0] : NAN;
    tc_tensor_nd_free(loss);
    tc_tensor_nd_free(out);
    return value;
}

static float maxpool_loss(tc_tensor_nd *x) {
    tc_tensor_nd *out = tc_tensor_nd_maxpool2d(x, 2, 2);
    tc_tensor_nd *loss = tc_tensor_nd_sum(out);
    const float value = loss ? tc_tensor_nd_const_data(loss)[0] : NAN;
    tc_tensor_nd_free(loss);
    tc_tensor_nd_free(out);
    return value;
}

static float avgpool_loss(tc_tensor_nd *x) {
    tc_tensor_nd *out = tc_tensor_nd_avgpool2d(x, 2, 2);
    tc_tensor_nd *loss = tc_tensor_nd_sum(out);
    const float value = loss ? tc_tensor_nd_const_data(loss)[0] : NAN;
    tc_tensor_nd_free(loss);
    tc_tensor_nd_free(out);
    return value;
}

static int test_reshape(void) {
    const size_t shape[] = {2, 2};
    const size_t flat[] = {4, 1};
    const float values[] = {1, 2, 3, 4};
    tc_tensor_nd *x = tc_tensor_nd_from_data(shape, 2, values, 1);
    tc_tensor_nd *flat_x = tc_tensor_nd_reshape_to(x, flat, 2);
    tc_tensor_nd *loss = tc_tensor_nd_sum(flat_x);
    int ok = x && flat_x && loss && tc_tensor_nd_ndim(flat_x) == 2 &&
             tc_tensor_nd_shape(flat_x)[0] == 4 &&
             tc_tensor_nd_shape(flat_x)[1] == 1 &&
             tc_tensor_nd_backward(loss);
    const float *out = tc_tensor_nd_const_data(flat_x);
    for (size_t i = 0; ok && i < 4; ++i)
        ok = close_enough(out[i], values[i]);
    /* Gradiente de sum(reshape(x)) = 1 en cada elemento. */
    const float *grad = tc_tensor_nd_grad(x);
    for (size_t i = 0; ok && i < 4; ++i)
        ok = close_enough(grad[i], 1.0f);
    ok = ok && finite_difference(x, reshape_loss, grad);

    /* Rechazo: número de elementos distinto. */
    const size_t bad[] = {3, 2};
    tc_tensor_nd *invalid = tc_tensor_nd_reshape_to(x, bad, 2);
    ok = ok && !invalid;
    tc_tensor_nd_free(invalid);
    tc_tensor_nd_free(loss);
    tc_tensor_nd_free(flat_x);
    tc_tensor_nd_free(x);
    return ok;
}

static int test_maxpool(void) {
    const size_t shape[] = {1, 1, 4, 4};
    const float values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    const float expected[] = {6, 8, 14, 16};
    tc_tensor_nd *x = tc_tensor_nd_from_data(shape, 4, values, 1);
    tc_tensor_nd *out = tc_tensor_nd_maxpool2d(x, 2, 2);
    int ok = x && out && tc_tensor_nd_ndim(out) == 4 &&
             tc_tensor_nd_shape(out)[2] == 2 && tc_tensor_nd_shape(out)[3] == 2;
    const float *data = tc_tensor_nd_const_data(out);
    for (size_t i = 0; ok && i < 4; ++i)
        ok = close_enough(data[i], expected[i]);

    /* Gradiente: fluye solo hacia el máximo de cada ventana. */
    tc_tensor_nd *loss = tc_tensor_nd_sum(out);
    ok = ok && loss && tc_tensor_nd_backward(loss);
    const float *grad = tc_tensor_nd_grad(x);
    for (size_t i = 0; ok && i < 16; ++i) {
        const int is_max = (i == 5) || (i == 7) || (i == 13) || (i == 15);
        ok = close_enough(grad[i], is_max ? 1.0f : 0.0f);
    }
    ok = ok && finite_difference(x, maxpool_loss, grad);

    /* Datos con máximos únicos y decimales para el chequeo numérico. */
    const size_t shape2[] = {1, 2, 4, 4};
    float values2[1 * 2 * 4 * 4];
    for (size_t i = 0; i < sizeof(values2) / sizeof(values2[0]); ++i)
        values2[i] = (float)((int)(i * 37 % 29) - 14) * 0.13f;
    tc_tensor_nd *x2 = tc_tensor_nd_from_data(shape2, 4, values2, 1);
    tc_tensor_nd *out2 = tc_tensor_nd_maxpool2d(x2, 2, 2);
    tc_tensor_nd *loss2 = tc_tensor_nd_sum(out2);
    ok = ok && x2 && out2 && loss2 && tc_tensor_nd_backward(loss2);
    ok = ok && finite_difference(x2, maxpool_loss, tc_tensor_nd_grad(x2));

    /* Rechazos: pool 0 y dimensiones no divisibles. */
    tc_tensor_nd *bad_pool = tc_tensor_nd_maxpool2d(x, 0, 2);
    tc_tensor_nd *bad_size = tc_tensor_nd_maxpool2d(x, 3, 2);
    ok = ok && !bad_pool && !bad_size;

    tc_tensor_nd_free(bad_size);
    tc_tensor_nd_free(bad_pool);
    tc_tensor_nd_free(loss2);
    tc_tensor_nd_free(out2);
    tc_tensor_nd_free(x2);
    tc_tensor_nd_free(loss);
    tc_tensor_nd_free(out);
    tc_tensor_nd_free(x);
    return ok;
}

static int test_layer(void) {
    /* Inicialización determinista: misma semilla, mismos pesos. */
    const tc_conv2d_options options = {1, 1, 1, 1};
    tc_conv2d_layer *a = tc_conv2d_layer_create(3, 4, 3, 3, options, 42);
    tc_conv2d_layer *b = tc_conv2d_layer_create(3, 4, 3, 3, options, 42);
    int ok = a && b;
    if (ok) {
        const float *ka = tc_tensor_nd_const_data(tc_conv2d_layer_kernel(a));
        const float *kb = tc_tensor_nd_const_data(tc_conv2d_layer_kernel(b));
        for (size_t i = 0; ok && i < tc_tensor_nd_numel(tc_conv2d_layer_kernel(a)); ++i)
            ok = ka[i] == kb[i];
        /* Sesgo inicializado en cero. */
        const float *bias = tc_tensor_nd_const_data(tc_conv2d_layer_bias(a));
        for (size_t i = 0; ok && i < tc_tensor_nd_numel(tc_conv2d_layer_bias(a)); ++i)
            ok = bias[i] == 0.0f;
    }

    /* Forward: forma correcta y sesgo agregado vía broadcasting. */
    const size_t input_shape[] = {2, 3, 6, 6};
    float input_values[2 * 3 * 6 * 6];
    for (size_t i = 0; i < sizeof(input_values) / sizeof(input_values[0]); ++i)
        input_values[i] = (float)((int)(i % 11) - 5) * 0.1f;
    tc_tensor_nd *input = tc_tensor_nd_from_data(input_shape, 4, input_values, 1);
    tc_tensor_nd *features = tc_conv2d_layer_forward(a, input);
    ok = ok && input && features && tc_tensor_nd_ndim(features) == 4 &&
         tc_tensor_nd_shape(features)[0] == 2 && tc_tensor_nd_shape(features)[1] == 4 &&
         tc_tensor_nd_shape(features)[2] == 6 && tc_tensor_nd_shape(features)[3] == 6;

    /* Con sesgo cero el forward debe coincidir con conv2d_ex directo. */
    if (ok) {
        tc_tensor_nd *bias = tc_conv2d_layer_bias(a);
        float *bias_data = tc_tensor_nd_data(bias);
        for (size_t i = 0; i < tc_tensor_nd_numel(bias); ++i)
            bias_data[i] = 0.25f;
        tc_tensor_nd *with_bias = tc_conv2d_layer_forward(a, input);
        tc_tensor_nd *plain = tc_tensor_nd_conv2d_ex(
            input, tc_conv2d_layer_kernel(a), options);
        ok = ok && with_bias && plain;
        if (ok) {
            const float *expected = tc_tensor_nd_const_data(plain);
            const float *actual = tc_tensor_nd_const_data(with_bias);
            for (size_t i = 0; ok && i < tc_tensor_nd_numel(with_bias); ++i)
                ok = close_enough(actual[i], expected[i] + 0.25f);
        }
        tc_tensor_nd_free(plain);
        tc_tensor_nd_free(with_bias);
    }

    /* Entrenamiento end-to-end: una capa aprende a reducir el MSE. */
    if (ok) {
        const size_t train_shape[] = {1, 3, 6, 6};
        const size_t target_shape[] = {1, 4, 6, 6};
        tc_tensor_nd *train_in = tc_tensor_nd_from_data(
            train_shape, 4, input_values, 1);
        float target_values[1 * 4 * 6 * 6];
        for (size_t i = 0; i < sizeof(target_values) / sizeof(target_values[0]); ++i)
            target_values[i] = sinf((float)i * 0.7f);
        tc_tensor_nd *target = tc_tensor_nd_from_data(
            target_shape, 4, target_values, 0);

        float first_loss = 0.0f, last_loss = 0.0f;
        for (int step = 0; ok && step < 60; ++step) {
            tc_tensor_nd *pred = tc_conv2d_layer_forward(a, train_in);
            tc_tensor_nd *loss = pred ? tc_tensor_nd_mse_loss(pred, target) : NULL;
            ok = ok && loss && tc_tensor_nd_backward(loss);
            if (step == 0) first_loss = tc_tensor_nd_const_data(loss)[0];
            last_loss = tc_tensor_nd_const_data(loss)[0];
            tc_tensor_nd_sgd_step(tc_conv2d_layer_kernel(a), 0.5f);
            tc_tensor_nd_sgd_step(tc_conv2d_layer_bias(a), 0.5f);
            tc_tensor_nd_free(loss);
            tc_tensor_nd_free(pred);
        }
        /* La prueba verifica el cableado del entrenamiento, no la
         * convergencia final: basta una reducción clara de la pérdida. */
        ok = ok && last_loss < first_loss * 0.6f;
        tc_tensor_nd_free(target);
        tc_tensor_nd_free(train_in);
    }

    tc_tensor_nd_free(features);
    tc_tensor_nd_free(input);
    tc_conv2d_layer_free(b);
    tc_conv2d_layer_free(a);
    return ok;
}

static int test_avgpool(void) {
    /* Valores conocidos: promedio de cada ventana 2x2. */
    const size_t shape[] = {1, 1, 4, 4};
    const float values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    const float expected[] = {3.5f, 5.5f, 11.5f, 13.5f};
    tc_tensor_nd *x = tc_tensor_nd_from_data(shape, 4, values, 1);
    tc_tensor_nd *out = tc_tensor_nd_avgpool2d(x, 2, 2);
    int ok = x && out && tc_tensor_nd_ndim(out) == 4 &&
             tc_tensor_nd_shape(out)[2] == 2 && tc_tensor_nd_shape(out)[3] == 2;
    const float *data = tc_tensor_nd_const_data(out);
    for (size_t i = 0; ok && i < 4; ++i)
        ok = close_enough(data[i], expected[i]);

    /* Gradiente: se reparte en partes iguales entre los 4 elementos de cada
     * ventana, así que con loss = sum(out) cada entrada vale 1/4. */
    tc_tensor_nd *loss = tc_tensor_nd_sum(out);
    ok = ok && loss && tc_tensor_nd_backward(loss);
    const float *grad = tc_tensor_nd_grad(x);
    for (size_t i = 0; ok && i < 16; ++i)
        ok = close_enough(grad[i], 0.25f);
    ok = ok && finite_difference(x, avgpool_loss, grad);

    /* Datos con decimales y varios canales para el chequeo numérico. */
    const size_t shape2[] = {1, 2, 4, 4};
    float values2[1 * 2 * 4 * 4];
    for (size_t i = 0; i < sizeof(values2) / sizeof(values2[0]); ++i)
        values2[i] = (float)((int)(i * 41 % 31) - 15) * 0.17f;
    tc_tensor_nd *x2 = tc_tensor_nd_from_data(shape2, 4, values2, 1);
    tc_tensor_nd *out2 = tc_tensor_nd_avgpool2d(x2, 2, 2);
    tc_tensor_nd *loss2 = tc_tensor_nd_sum(out2);
    ok = ok && x2 && out2 && loss2 && tc_tensor_nd_backward(loss2);
    ok = ok && finite_difference(x2, avgpool_loss, tc_tensor_nd_grad(x2));

    /* Rechazos: pool 0 y dimensiones no divisibles. */
    tc_tensor_nd *bad_pool = tc_tensor_nd_avgpool2d(x, 0, 2);
    tc_tensor_nd *bad_size = tc_tensor_nd_avgpool2d(x, 3, 2);
    ok = ok && !bad_pool && !bad_size;

    tc_tensor_nd_free(bad_size);
    tc_tensor_nd_free(bad_pool);
    tc_tensor_nd_free(loss2);
    tc_tensor_nd_free(out2);
    tc_tensor_nd_free(x2);
    tc_tensor_nd_free(loss);
    tc_tensor_nd_free(out);
    tc_tensor_nd_free(x);
    return ok;
}

int main(void) {
    const int reshape_ok = test_reshape();
    const int maxpool_ok = test_maxpool();
    const int avgpool_ok = test_avgpool();
    const int layer_ok = test_layer();
    printf("reshape: %s\n", reshape_ok ? "ok" : "error");
    printf("maxpool: %s\n", maxpool_ok ? "ok" : "error");
    printf("avgpool: %s\n", avgpool_ok ? "ok" : "error");
    printf("layer: %s\n", layer_ok ? "ok" : "error");
    return (reshape_ok && maxpool_ok && avgpool_ok && layer_ok) ? 0 : 1;
}
