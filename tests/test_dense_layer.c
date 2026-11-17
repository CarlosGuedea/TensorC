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

static float sigmoid_loss(tc_tensor_nd *x) {
    tc_tensor_nd *out = tc_tensor_nd_sigmoid(x);
    tc_tensor_nd *loss = tc_tensor_nd_sum(out);
    const float value = loss ? tc_tensor_nd_const_data(loss)[0] : NAN;
    tc_tensor_nd_free(loss);
    tc_tensor_nd_free(out);
    return value;
}

static float tanh_loss(tc_tensor_nd *x) {
    tc_tensor_nd *out = tc_tensor_nd_tanh(x);
    tc_tensor_nd *loss = tc_tensor_nd_sum(out);
    const float value = loss ? tc_tensor_nd_const_data(loss)[0] : NAN;
    tc_tensor_nd_free(loss);
    tc_tensor_nd_free(out);
    return value;
}

static int test_sigmoid(void) {
    const size_t shape[] = {2, 3};
    /* Valores positivos, negativos y el punto medio. */
    const float values[] = {-4.0f, -1.0f, 0.0f, 0.5f, 2.0f, 8.0f};
    tc_tensor_nd *x = tc_tensor_nd_from_data(shape, 2, values, 1);
    tc_tensor_nd *out = tc_tensor_nd_sigmoid(x);
    int ok = x && out && tc_tensor_nd_ndim(out) == 2 &&
             tc_tensor_nd_shape(out)[0] == 2 && tc_tensor_nd_shape(out)[1] == 3;
    const float *data = tc_tensor_nd_const_data(out);
    for (size_t i = 0; ok && i < 6; ++i) {
        const float expected = 1.0f / (1.0f + expf(-values[i]));
        ok = close_enough(data[i], expected);
    }
    /* Estabilidad numérica: entradas extremas no producen NaN/inf. */
    const size_t extreme_shape[] = {2};
    const float extreme[] = {-90.0f, 90.0f};
    tc_tensor_nd *x2 = tc_tensor_nd_from_data(extreme_shape, 1, extreme, 0);
    tc_tensor_nd *out2 = tc_tensor_nd_sigmoid(x2);
    ok = ok && out2 && isfinite(tc_tensor_nd_const_data(out2)[0]) &&
         isfinite(tc_tensor_nd_const_data(out2)[1]) &&
         close_enough(tc_tensor_nd_const_data(out2)[0], 0.0f) &&
         close_enough(tc_tensor_nd_const_data(out2)[1], 1.0f);

    /* Gradiente: d sigmoid = y * (1 - y). */
    tc_tensor_nd *loss = tc_tensor_nd_sum(out);
    ok = ok && loss && tc_tensor_nd_backward(loss);
    const float *grad = tc_tensor_nd_grad(x);
    for (size_t i = 0; ok && i < 6; ++i) {
        const float expected = data[i] * (1.0f - data[i]);
        ok = close_enough(grad[i], expected);
    }
    ok = ok && finite_difference(x, sigmoid_loss, grad);

    tc_tensor_nd_free(loss);
    tc_tensor_nd_free(out);
    tc_tensor_nd_free(x);
    tc_tensor_nd_free(out2);
    tc_tensor_nd_free(x2);
    return ok;
}

static int test_tanh(void) {
    const size_t shape[] = {2, 3};
    const float values[] = {-2.0f, -0.5f, 0.0f, 0.25f, 1.0f, 3.0f};
    tc_tensor_nd *x = tc_tensor_nd_from_data(shape, 2, values, 1);
    tc_tensor_nd *out = tc_tensor_nd_tanh(x);
    int ok = x && out && tc_tensor_nd_ndim(out) == 2 &&
             tc_tensor_nd_shape(out)[0] == 2 && tc_tensor_nd_shape(out)[1] == 3;
    const float *data = tc_tensor_nd_const_data(out);
    for (size_t i = 0; ok && i < 6; ++i)
        ok = close_enough(data[i], tanhf(values[i]));

    /* Gradiente: d tanh = 1 - y^2. */
    tc_tensor_nd *loss = tc_tensor_nd_sum(out);
    ok = ok && loss && tc_tensor_nd_backward(loss);
    const float *grad = tc_tensor_nd_grad(x);
    for (size_t i = 0; ok && i < 6; ++i)
        ok = close_enough(grad[i], 1.0f - data[i] * data[i]);
    ok = ok && finite_difference(x, tanh_loss, grad);

    tc_tensor_nd_free(loss);
    tc_tensor_nd_free(out);
    tc_tensor_nd_free(x);
    return ok;
}

static int test_dense_layer(void) {
    /* Inicialización determinista: misma semilla, mismos pesos. */
    tc_dense_layer *a = tc_dense_layer_create(4, 3, 42);
    tc_dense_layer *b = tc_dense_layer_create(4, 3, 42);
    int ok = a && b;
    if (ok) {
        const float *wa = tc_tensor_nd_const_data(tc_dense_layer_weight(a));
        const float *wb = tc_tensor_nd_const_data(tc_dense_layer_weight(b));
        for (size_t i = 0; ok && i < tc_tensor_nd_numel(tc_dense_layer_weight(a)); ++i)
            ok = wa[i] == wb[i];
        /* Sesgo inicializado en cero. */
        const float *bias = tc_tensor_nd_const_data(tc_dense_layer_bias(a));
        for (size_t i = 0; ok && i < tc_tensor_nd_numel(tc_dense_layer_bias(a)); ++i)
            ok = bias[i] == 0.0f;
    }

    /* Forward: forma correcta y valores con pesos/sesgo conocidos. */
    const size_t input_shape[] = {2, 4};
    const float input_values[] = {1, 2, 3, 4, -1, 0.5f, 2, -3};
    tc_tensor_nd *input = tc_tensor_nd_from_data(input_shape, 2, input_values, 1);
    tc_tensor_nd *features = tc_dense_layer_forward(a, input);
    ok = ok && input && features && tc_tensor_nd_ndim(features) == 2 &&
         tc_tensor_nd_shape(features)[0] == 2 && tc_tensor_nd_shape(features)[1] == 3;

    if (ok) {
        float *w = tc_tensor_nd_data(tc_dense_layer_weight(a));
        float *bias_data = tc_tensor_nd_data(tc_dense_layer_bias(a));
        /* Pesos de identidad expandidos a 3 salidas + sesgo conocido. */
        for (size_t i = 0; i < 12; ++i) w[i] = 0.0f;
        for (size_t r = 0; r < 4; ++r) {
            w[r * 3 + 0] = 1.0f;   /* salida 0 = suma de entradas */
            w[r * 3 + 1] = 0.5f;   /* salida 1 = media de entradas */
        }
        /* salida 2 queda en cero + sesgo. */
        bias_data[0] = 0.1f;
        bias_data[1] = -0.2f;
        bias_data[2] = 2.0f;
        tc_tensor_nd *out = tc_dense_layer_forward(a, input);
        ok = ok && out;
        if (ok) {
            const float *y = tc_tensor_nd_const_data(out);
            /* Fila 0: entradas 1,2,3,4 -> suma 10, media 5. */
            ok = close_enough(y[0], 10.1f) && close_enough(y[1], 4.8f) &&
                 close_enough(y[2], 2.0f);
            /* Fila 1: entradas -1, 0.5, 2, -3 -> suma -1.5, media -0.75. */
            ok = ok && close_enough(y[3], -1.4f) && close_enough(y[4], -0.95f) &&
                 close_enough(y[5], 2.0f);
        }
        tc_tensor_nd_free(out);

        /* Gradiente: sum(forward) debe propagar a entrada, pesos y sesgo. */
        tc_tensor_nd *loss = tc_dense_layer_forward(a, input);
        tc_tensor_nd *loss_sum = loss ? tc_tensor_nd_sum(loss) : NULL;
        ok = ok && loss_sum && tc_tensor_nd_backward(loss_sum);
        if (ok) {
            /* Cada gradiente de entrada es la suma de las columnas de W
             * (salida 0: 1, salida 1: 0.5, salida 2: 0 -> total 1.5). */
            const float *grad_in = tc_tensor_nd_grad(input);
            for (size_t r = 0; ok && r < 4; ++r)
                ok = close_enough(grad_in[r], 1.5f);
            /* Gradientes de peso: con upstream=1, grad_w[r][c] es la suma de
             * la columna r de la entrada, igual para toda columna c de
             * salida (el gradiente no depende del valor del peso). */
            const float *grad_w = tc_tensor_nd_grad(tc_dense_layer_weight(a));
            const float col_sums[] = {0.0f, 2.5f, 5.0f, 1.0f};
            for (size_t r = 0; ok && r < 4; ++r) {
                ok = close_enough(grad_w[r * 3 + 0], col_sums[r]) &&
                     close_enough(grad_w[r * 3 + 1], col_sums[r]) &&
                     close_enough(grad_w[r * 3 + 2], col_sums[r]);
            }
            /* Gradiente de sesgo: una fila por salida. */
            const float *grad_b = tc_tensor_nd_grad(tc_dense_layer_bias(a));
            ok = ok && close_enough(grad_b[0], 2.0f) &&
                 close_enough(grad_b[1], 2.0f) && close_enough(grad_b[2], 2.0f);
        }
        tc_tensor_nd_free(loss_sum);
        tc_tensor_nd_free(loss);
    }

    /* Rechazos: entrada de rango distinto de 2 y dimensión incompatible. */
    const size_t bad_shape[] = {2, 2, 4};
    float bad_values[2 * 2 * 4];
    for (size_t i = 0; i < sizeof(bad_values) / sizeof(bad_values[0]); ++i)
        bad_values[i] = (float)(i % 5) * 0.1f;
    tc_tensor_nd *bad_input = tc_tensor_nd_from_data(bad_shape, 3, bad_values, 0);
    tc_tensor_nd *bad_out = tc_dense_layer_forward(a, bad_input);
    const size_t wrong_shape[] = {2, 5};
    float wrong_values[2 * 5];
    for (size_t i = 0; i < sizeof(wrong_values) / sizeof(wrong_values[0]); ++i)
        wrong_values[i] = (float)(i % 7) * 0.1f;
    tc_tensor_nd *wrong_input = tc_tensor_nd_from_data(wrong_shape, 2, wrong_values, 0);
    tc_tensor_nd *wrong_out = tc_dense_layer_forward(a, wrong_input);
    tc_dense_layer *null_layer = tc_dense_layer_create(0, 3, 1);
    ok = ok && !bad_out && !wrong_out && !null_layer;

    /* Entrenamiento end-to-end: la capa aprende y = 2x - 1. */
    if (ok) {
        tc_dense_layer *lin = tc_dense_layer_create(1, 1, 7);
        ok = ok && lin;
        if (ok) {
            const size_t xs_shape[] = {4, 1};
            const float xs[] = {-2, -1, 0, 3};
            tc_tensor_nd *xs_t = tc_tensor_nd_from_data(xs_shape, 2, xs, 0);
            float ys[4];
            for (size_t i = 0; i < 4; ++i) ys[i] = 2.0f * xs[i] - 1.0f;
            const size_t ys_shape[] = {4, 1};
            tc_tensor_nd *ys_t = tc_tensor_nd_from_data(ys_shape, 2, ys, 0);

            float first_loss = 0.0f, last_loss = 0.0f;
            for (int step = 0; ok && step < 200; ++step) {
                tc_tensor_nd *pred = tc_dense_layer_forward(lin, xs_t);
                tc_tensor_nd *loss = pred ? tc_tensor_nd_mse_loss(pred, ys_t) : NULL;
                ok = ok && loss && tc_tensor_nd_backward(loss);
                if (step == 0) first_loss = tc_tensor_nd_const_data(loss)[0];
                last_loss = tc_tensor_nd_const_data(loss)[0];
                tc_tensor_nd_sgd_step(tc_dense_layer_weight(lin), 0.05f);
                tc_tensor_nd_sgd_step(tc_dense_layer_bias(lin), 0.05f);
                tc_tensor_nd_free(loss);
                tc_tensor_nd_free(pred);
            }
            ok = ok && last_loss < first_loss * 0.05f;
            tc_tensor_nd_free(ys_t);
            tc_tensor_nd_free(xs_t);
        }
        tc_dense_layer_free(lin);
    }

    tc_tensor_nd_free(wrong_out);
    tc_tensor_nd_free(wrong_input);
    tc_tensor_nd_free(bad_out);
    tc_tensor_nd_free(bad_input);
    tc_tensor_nd_free(features);
    tc_tensor_nd_free(input);
    tc_dense_layer_free(b);
    tc_dense_layer_free(a);
    return ok;
}

int main(void) {
    const int sigmoid_ok = test_sigmoid();
    const int tanh_ok = test_tanh();
    const int layer_ok = test_dense_layer();
    printf("sigmoid: %s\n", sigmoid_ok ? "ok" : "error");
    printf("tanh: %s\n", tanh_ok ? "ok" : "error");
    printf("dense_layer: %s\n", layer_ok ? "ok" : "error");
    return (sigmoid_ok && tanh_ok && layer_ok) ? 0 : 1;
}
