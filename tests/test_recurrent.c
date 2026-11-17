#include <tensorc/tensorc.h>

#include <math.h>
#include <stdio.h>

static int close_enough(float actual, float expected) {
    return fabsf(actual - expected) < 1e-4f;
}

static int gradient_close(float analytic, float numeric) {
    return fabsf(analytic - numeric) <= 2e-3f * (1.0f + fabsf(numeric));
}

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
        if (!gradient_close(analytic[i], numeric)) {
            fprintf(stderr, "  idx %zu: analitico=%f numerico=%f\n", i,
                    analytic[i], numeric);
            return 0;
        }
    }
    return 1;
}

static float slice_loss(tc_tensor_nd *x) {
    tc_tensor_nd *out = tc_tensor_nd_slice_axis(x, 0, 1, 2);
    tc_tensor_nd *loss = out ? tc_tensor_nd_sum(out) : NULL;
    const float value = loss ? tc_tensor_nd_const_data(loss)[0] : NAN;
    tc_tensor_nd_free(loss);
    tc_tensor_nd_free(out);
    return value;
}

static float layernorm_loss(tc_tensor_nd *x) {
    tc_tensor_nd *out = tc_tensor_nd_layer_norm(x);
    tc_tensor_nd *loss = out ? tc_tensor_nd_sum(out) : NULL;
    const float value = loss ? tc_tensor_nd_const_data(loss)[0] : NAN;
    tc_tensor_nd_free(loss);
    tc_tensor_nd_free(out);
    return value;
}

static float transpose_loss(tc_tensor_nd *x) {
    tc_tensor_nd *out = tc_tensor_nd_transpose(x);
    tc_tensor_nd *loss = out ? tc_tensor_nd_sum(out) : NULL;
    const float value = loss ? tc_tensor_nd_const_data(loss)[0] : NAN;
    tc_tensor_nd_free(loss);
    tc_tensor_nd_free(out);
    return value;
}

/* loss(q) = sum(attention(q, k, v)); k y v fijos en globals. */
static tc_tensor_nd *g_k, *g_v;
static float attention_loss(tc_tensor_nd *q) {
    tc_tensor_nd *out = tc_tensor_nd_attention(q, g_k, g_v, 0);
    tc_tensor_nd *loss = out ? tc_tensor_nd_sum(out) : NULL;
    const float value = loss ? tc_tensor_nd_const_data(loss)[0] : NAN;
    tc_tensor_nd_free(loss);
    tc_tensor_nd_free(out);
    return value;
}

static int test_slice_concat(void) {
    const size_t shape[] = {4, 3};
    const float values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    tc_tensor_nd *x = tc_tensor_nd_from_data(shape, 2, values, 1);

    /* Slice en el eje 0: filas 1..2. */
    tc_tensor_nd *s = tc_tensor_nd_slice_axis(x, 0, 1, 2);
    int ok = x && s && tc_tensor_nd_ndim(s) == 2 &&
             tc_tensor_nd_shape(s)[0] == 2 && tc_tensor_nd_shape(s)[1] == 3;
    const float expected[] = {4, 5, 6, 7, 8, 9};
    const float *sd = tc_tensor_nd_const_data(s);
    for (size_t i = 0; ok && i < 6; ++i) ok = close_enough(sd[i], expected[i]);

    /* Gradiente: fluye solo a las filas 1..2. */
    tc_tensor_nd *loss = tc_tensor_nd_sum(s);
    ok = ok && loss && tc_tensor_nd_backward(loss);
    const float *grad = tc_tensor_nd_grad(x);
    for (size_t i = 0; ok && i < 12; ++i) {
        const int in_slice = (i >= 3 && i < 9);
        ok = close_enough(grad[i], in_slice ? 1.0f : 0.0f);
    }
    ok = ok && finite_difference(x, slice_loss, grad);

    /* Slice en el eje 1: columna 2. */
    tc_tensor_nd *s2 = tc_tensor_nd_slice_axis(x, 1, 2, 1);
    ok = ok && s2 && tc_tensor_nd_shape(s2)[0] == 4 &&
         tc_tensor_nd_shape(s2)[1] == 1;
    const float *s2d = tc_tensor_nd_const_data(s2);
    ok = ok && close_enough(s2d[0], 3.0f) && close_enough(s2d[1], 6.0f) &&
         close_enough(s2d[2], 9.0f) && close_enough(s2d[3], 12.0f);

    /* Concat de dos slices recupera el original (eje 0). */
    tc_tensor_nd *a = tc_tensor_nd_slice_axis(x, 0, 0, 2);
    tc_tensor_nd *b = tc_tensor_nd_slice_axis(x, 0, 2, 2);
    const tc_tensor_nd *parts[] = {a, b};
    tc_tensor_nd *joined = tc_tensor_nd_concat(parts, 2, 0);
    ok = ok && joined && tc_tensor_nd_shape(joined)[0] == 4;
    const float *jd = tc_tensor_nd_const_data(joined);
    for (size_t i = 0; ok && i < 12; ++i) ok = close_enough(jd[i], values[i]);

    /* Gradiente del concat: cada parte recibe su tramo. */
    tc_tensor_nd *closs = tc_tensor_nd_sum(joined);
    ok = ok && closs && tc_tensor_nd_backward(closs);
    const float *ag = tc_tensor_nd_grad(a);
    const float *bg = tc_tensor_nd_grad(b);
    ok = ok && ag && bg && close_enough(ag[0], 1.0f) &&
         close_enough(ag[5], 1.0f) && close_enough(bg[0], 1.0f) &&
         close_enough(bg[5], 1.0f);

    /* Rechazos: rango inválido, start fuera de rango, count 0. */
    ok = ok && !tc_tensor_nd_slice_axis(x, 2, 0, 1) &&
         !tc_tensor_nd_slice_axis(x, 0, 3, 2) &&
         !tc_tensor_nd_slice_axis(x, 0, 0, 0) &&
         !tc_tensor_nd_concat(parts, 0, 0);

    tc_tensor_nd_free(closs);
    tc_tensor_nd_free(joined);
    tc_tensor_nd_free(a);
    tc_tensor_nd_free(b);
    tc_tensor_nd_free(loss);
    tc_tensor_nd_free(s2);
    tc_tensor_nd_free(s);
    tc_tensor_nd_free(x);
    return ok;
}

static int test_layer_norm(void) {
    const size_t shape[] = {2, 4};
    const float values[] = {1, 3, 5, 9, -4, 0, 2, 6};
    tc_tensor_nd *x = tc_tensor_nd_from_data(shape, 2, values, 1);
    tc_tensor_nd *out = tc_tensor_nd_layer_norm(x);
    int ok = x && out && tc_tensor_nd_ndim(out) == 2 &&
             tc_tensor_nd_shape(out)[0] == 2 && tc_tensor_nd_shape(out)[1] == 4;

    /* Verificar media ~0 y varianza ~1 por fila, y contra cálculo manual. */
    const float *od = tc_tensor_nd_const_data(out);
    for (size_t row = 0; ok && row < 2; ++row) {
        float mean = 0.0f, var = 0.0f;
        for (size_t j = 0; j < 4; ++j) mean += od[row * 4 + j];
        mean /= 4.0f;
        for (size_t j = 0; j < 4; ++j)
            var += (od[row * 4 + j] - mean) * (od[row * 4 + j] - mean);
        var /= 4.0f;
        ok = close_enough(mean, 0.0f) && close_enough(var, 1.0f);
        /* Manual: */
        float m2 = 0.0f;
        for (size_t j = 0; j < 4; ++j) m2 += values[row * 4 + j];
        m2 /= 4.0f;
        float v2 = 0.0f;
        for (size_t j = 0; j < 4; ++j)
            v2 += (values[row * 4 + j] - m2) * (values[row * 4 + j] - m2);
        v2 /= 4.0f;
        for (size_t j = 0; ok && j < 4; ++j)
            ok = close_enough(od[row * 4 + j],
                              (values[row * 4 + j] - m2) / sqrtf(v2 + 1e-5f));
    }

    /* Gradiente por diferencias finitas. */
    tc_tensor_nd *loss = tc_tensor_nd_sum(out);
    ok = ok && loss && tc_tensor_nd_backward(loss);
    ok = ok && finite_difference(x, layernorm_loss, tc_tensor_nd_grad(x));

    /* Rechazo: tensor de rango 0 (no aplicable). */
    ok = ok && !tc_tensor_nd_layer_norm(NULL);

    tc_tensor_nd_free(loss);
    tc_tensor_nd_free(out);
    tc_tensor_nd_free(x);
    return ok;
}

static int test_transpose(void) {
    const size_t shape[] = {2, 3};
    const float values[] = {1, 2, 3, 4, 5, 6};
    tc_tensor_nd *x = tc_tensor_nd_from_data(shape, 2, values, 1);
    tc_tensor_nd *t = tc_tensor_nd_transpose(x);
    int ok = x && t && tc_tensor_nd_shape(t)[0] == 3 &&
             tc_tensor_nd_shape(t)[1] == 2;
    const float expected[] = {1, 4, 2, 5, 3, 6};
    const float *td = tc_tensor_nd_const_data(t);
    for (size_t i = 0; ok && i < 6; ++i) ok = close_enough(td[i], expected[i]);

    /* doble transpuesta = identidad */
    tc_tensor_nd *tt = tc_tensor_nd_transpose(t);
    ok = ok && tt;
    const float *ttd = tc_tensor_nd_const_data(tt);
    for (size_t i = 0; ok && i < 6; ++i) ok = close_enough(ttd[i], values[i]);

    tc_tensor_nd *loss = tc_tensor_nd_sum(t);
    ok = ok && loss && tc_tensor_nd_backward(loss);
    ok = ok && finite_difference(x, transpose_loss, tc_tensor_nd_grad(x));
    ok = ok && !tc_tensor_nd_transpose(NULL);

    tc_tensor_nd_free(loss);
    tc_tensor_nd_free(tt);
    tc_tensor_nd_free(t);
    tc_tensor_nd_free(x);
    return ok;
}

static int test_attention(void) {
    /* Q=K=V con valores distintos por posición para distinguir pesos. */
    const size_t shape[] = {1, 3, 2};
    const float qv[] = {1, 0, 0, 1, 1, 1};
    const float kv[] = {1, 2, 0, 1, 2, 0};
    const float vv[] = {0.5f, 1.5f, -1.0f, 2.0f, 3.0f, -2.0f};
    tc_tensor_nd *q = tc_tensor_nd_from_data(shape, 3, qv, 1);
    tc_tensor_nd *k = tc_tensor_nd_from_data(shape, 3, kv, 0);
    tc_tensor_nd *v = tc_tensor_nd_from_data(shape, 3, vv, 0);
    g_k = k;
    g_v = v;

    /* Referencia manual en C (implementación independiente). */
    const float scale = 1.0f / sqrtf(2.0f);
    float ref[3 * 2];
    for (size_t i = 0; i < 3; ++i) {
        float scores[3];
        for (size_t j = 0; j < 3; ++j) {
            float dot = 0.0f;
            for (size_t d = 0; d < 2; ++d)
                dot += qv[i * 2 + d] * kv[j * 2 + d];
            scores[j] = dot * scale;
        }
        float mx = scores[0];
        for (size_t j = 1; j < 3; ++j) if (scores[j] > mx) mx = scores[j];
        float sum = 0.0f;
        for (size_t j = 0; j < 3; ++j) {
            scores[j] = expf(scores[j] - mx);
            sum += scores[j];
        }
        for (size_t d = 0; d < 2; ++d) {
            float acc = 0.0f;
            for (size_t j = 0; j < 3; ++j)
                acc += (scores[j] / sum) * vv[j * 2 + d];
            ref[i * 2 + d] = acc;
        }
    }

    tc_tensor_nd *out = tc_tensor_nd_attention(q, k, v, 0);
    int ok = q && k && v && out && tc_tensor_nd_ndim(out) == 3 &&
             tc_tensor_nd_shape(out)[0] == 1 && tc_tensor_nd_shape(out)[1] == 3 &&
             tc_tensor_nd_shape(out)[2] == 2;
    const float *od = tc_tensor_nd_const_data(out);
    for (size_t i = 0; ok && i < 6; ++i) ok = close_enough(od[i], ref[i]);

    /* Gradiente de Q por diferencias finitas (K, V fijos). */
    tc_tensor_nd *loss = tc_tensor_nd_sum(out);
    ok = ok && loss && tc_tensor_nd_backward(loss);
    ok = ok && finite_difference(q, attention_loss, tc_tensor_nd_grad(q));

    /* Variante causal: la fila 0 solo ve la posición 0, luego su salida
     * debe ser exactamente V[0]. */
    tc_tensor_nd *cout = tc_tensor_nd_attention(q, k, v, 1);
    ok = ok && cout;
    const float *cd = tc_tensor_nd_const_data(cout);
    ok = ok && close_enough(cd[0], 0.5f) && close_enough(cd[1], 1.5f);

    /* Rechazos: depth distinto entre Q y K, rango != 3. */
    const size_t bad_shape[] = {1, 3, 4};
    float bad_data[12] = {0};
    tc_tensor_nd *bad_k = tc_tensor_nd_from_data(bad_shape, 3, bad_data, 0);
    const size_t r2[] = {3, 2};
    tc_tensor_nd *bad_r = tc_tensor_nd_from_data(r2, 2, qv, 0);
    ok = ok && !tc_tensor_nd_attention(q, bad_k, v, 0) &&
         !tc_tensor_nd_attention(bad_r, bad_r, bad_r, 0);

    tc_tensor_nd_free(bad_r);
    tc_tensor_nd_free(bad_k);
    tc_tensor_nd_free(cout);
    tc_tensor_nd_free(loss);
    tc_tensor_nd_free(out);
    tc_tensor_nd_free(v);
    tc_tensor_nd_free(k);
    tc_tensor_nd_free(q);
    return ok;
}

/* Referencia LSTM en C puro para un paso (implementación independiente
 * usada solo por la prueba). */
static void lstm_step_ref(const float *x, const float *h, const float *c,
                          const float *w_ih, const float *w_hh,
                          const float *bias, size_t input, size_t hidden,
                          float *h_out, float *c_out) {
    float gates[64];
    for (size_t g = 0; g < 4 * hidden; ++g) {
        float acc = bias[g];
        for (size_t i = 0; i < input; ++i) acc += w_ih[g * input + i] * x[i];
        for (size_t i = 0; i < hidden; ++i) acc += w_hh[g * hidden + i] * h[i];
        gates[g] = acc;
    }
    for (size_t i = 0; i < hidden; ++i) {
        const float ig = 1.0f / (1.0f + expf(-gates[i]));
        const float fg = 1.0f / (1.0f + expf(-gates[hidden + i]));
        const float gg = tanhf(gates[2 * hidden + i]);
        const float og = 1.0f / (1.0f + expf(-gates[3 * hidden + i]));
        c_out[i] = fg * c[i] + ig * gg;
        h_out[i] = og * tanhf(c_out[i]);
    }
}

static int test_lstm(void) {
    /* Inicialización determinista y sesgo cero. */
    tc_lstm_layer *a = tc_lstm_layer_create(2, 3, 42);
    tc_lstm_layer *b = tc_lstm_layer_create(2, 3, 42);
    int ok = a && b;
    if (ok) {
        const float *wa = tc_tensor_nd_const_data(tc_lstm_layer_weight_ih(a));
        const float *wb = tc_tensor_nd_const_data(tc_lstm_layer_weight_ih(b));
        for (size_t i = 0; ok && i < tc_tensor_nd_numel(tc_lstm_layer_weight_ih(a)); ++i)
            ok = wa[i] == wb[i];
        const float *bias = tc_tensor_nd_const_data(tc_lstm_layer_bias(a));
        for (size_t i = 0; ok && i < tc_tensor_nd_numel(tc_lstm_layer_bias(a)); ++i)
            ok = bias[i] == 0.0f;
    }

    /* Forward contra la referencia paso a paso (seq=3, batch=1). */
    const size_t shape[] = {3, 1, 2};
    const float xv[] = {0.5f, -0.25f, 1.0f, 2.0f, -1.5f, 0.75f};
    tc_tensor_nd *x = tc_tensor_nd_from_data(shape, 3, xv, 1);
    tc_tensor_nd *out = tc_lstm_layer_forward(a, x);
    ok = ok && x && out && tc_tensor_nd_ndim(out) == 3 &&
         tc_tensor_nd_shape(out)[0] == 3 && tc_tensor_nd_shape(out)[1] == 1 &&
         tc_tensor_nd_shape(out)[2] == 3;

    if (ok) {
        const float *w_ih = tc_tensor_nd_const_data(tc_lstm_layer_weight_ih(a));
        const float *w_hh = tc_tensor_nd_const_data(tc_lstm_layer_weight_hh(a));
        const float *bias = tc_tensor_nd_const_data(tc_lstm_layer_bias(a));
        float h[3] = {0, 0, 0}, c[3] = {0, 0, 0};
        const float *od = tc_tensor_nd_const_data(out);
        for (size_t t = 0; ok && t < 3; ++t) {
            lstm_step_ref(xv + t * 2, h, c, w_ih, w_hh, bias, 2, 3, h, c);
            for (size_t j = 0; ok && j < 3; ++j)
                ok = close_enough(od[(t * 1 + 0) * 3 + j], h[j]);
        }
    }

    /* backward corre sobre toda la secuencia y los gradientes llegan a
     * pesos y entrada. */
    tc_tensor_nd *loss = tc_tensor_nd_sum(out);
    ok = ok && loss && tc_tensor_nd_backward(loss);
    ok = ok && tc_tensor_nd_grad(tc_lstm_layer_weight_ih(a)) &&
         tc_tensor_nd_grad(tc_lstm_layer_weight_hh(a)) &&
         tc_tensor_nd_grad(x);
    if (ok) {
        const float *g = tc_tensor_nd_grad(tc_lstm_layer_weight_ih(a));
        for (size_t i = 0; ok && i < tc_tensor_nd_numel(tc_lstm_layer_weight_ih(a)); ++i)
            ok = isfinite(g[i]);
        const float *gx = tc_tensor_nd_grad(x);
        for (size_t i = 0; ok && i < 6; ++i) ok = isfinite(gx[i]);
    }

    /* Entrenamiento end-to-end: el LSTM aprende a copiar la entrada en
     * su salida (hidden = input = 1, salida [seq, batch, 1]). */
    if (ok) {
        tc_lstm_layer *net = tc_lstm_layer_create(1, 1, 7);
        const size_t xs_shape[] = {4, 1, 1};
        const float xs[] = {0.3f, -0.6f, 0.9f, -0.2f};
        const float ys[] = {0.3f, -0.6f, 0.9f, -0.2f};
        tc_tensor_nd *xs_t = tc_tensor_nd_from_data(xs_shape, 3, xs, 0);
        tc_tensor_nd *ys_t = tc_tensor_nd_from_data(xs_shape, 3, ys, 0);
        tc_tensor_nd *params[] = {
            tc_lstm_layer_weight_ih(net), tc_lstm_layer_weight_hh(net),
            tc_lstm_layer_bias(net)};
        tc_tensor_nd_optimizer *adam = tc_tensor_nd_optimizer_adam(params, 3, 0.05f);

        float first_loss = 0.0f, last_loss = 0.0f;
        for (int step = 0; ok && step < 120; ++step) {
            tc_tensor_nd *pred = tc_lstm_layer_forward(net, xs_t);
            tc_tensor_nd *l = pred ? tc_tensor_nd_mse_loss(pred, ys_t) : NULL;
            ok = ok && l && tc_tensor_nd_backward(l);
            if (step == 0) first_loss = tc_tensor_nd_const_data(l)[0];
            last_loss = tc_tensor_nd_const_data(l)[0];
            tc_tensor_nd_optimizer_step(adam);
            tc_tensor_nd_free(l);
            tc_tensor_nd_free(pred);
        }
        ok = ok && last_loss < first_loss * 0.5f;
        tc_tensor_nd_optimizer_free(adam);
        tc_tensor_nd_free(ys_t);
        tc_tensor_nd_free(xs_t);
        tc_lstm_layer_free(net);
    }

    /* Rechazos: entrada de rango distinto de 3 y features incompatibles. */
    const size_t bad3[] = {3, 1, 5};
    float bad3d[15] = {0};
    tc_tensor_nd *bad_in = tc_tensor_nd_from_data(bad3, 3, bad3d, 0);
    ok = ok && !tc_lstm_layer_forward(a, bad_in) &&
         !tc_lstm_layer_create(0, 3, 1);
    tc_tensor_nd_free(bad_in);

    tc_tensor_nd_free(loss);
    tc_tensor_nd_free(out);
    tc_tensor_nd_free(x);
    tc_lstm_layer_free(b);
    tc_lstm_layer_free(a);
    return ok;
}

int main(void) {
    const int slice_concat_ok = test_slice_concat();
    const int layernorm_ok = test_layer_norm();
    const int transpose_ok = test_transpose();
    const int attention_ok = test_attention();
    const int lstm_ok = test_lstm();
    printf("slice_concat: %s\n", slice_concat_ok ? "ok" : "error");
    printf("layer_norm: %s\n", layernorm_ok ? "ok" : "error");
    printf("transpose: %s\n", transpose_ok ? "ok" : "error");
    printf("attention: %s\n", attention_ok ? "ok" : "error");
    printf("lstm: %s\n", lstm_ok ? "ok" : "error");
    return (slice_concat_ok && layernorm_ok && transpose_ok && attention_ok &&
            lstm_ok) ? 0 : 1;
}
