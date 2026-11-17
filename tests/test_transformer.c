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
    const float epsilon = 1e-3f;
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

static tc_tensor_nd *g_perm_w = NULL;

/* loss = sum(permute(x) * w): el gradiente queda condicionado por w y
 * además ejercita el mapeo de la permutación (no solo su invariancia). */
static float permute_loss(tc_tensor_nd *x) {
    const size_t axes[] = {2, 0, 1};
    tc_tensor_nd *out = tc_tensor_nd_permute(x, axes, 3);
    tc_tensor_nd *m = out ? tc_tensor_nd_mul(out, g_perm_w) : NULL;
    tc_tensor_nd *loss = m ? tc_tensor_nd_sum(m) : NULL;
    const float value = loss ? tc_tensor_nd_const_data(loss)[0] : NAN;
    tc_tensor_nd_free(loss);
    tc_tensor_nd_free(m);
    tc_tensor_nd_free(out);
    return value;
}

static float multihead_loss(tc_tensor_nd *q) {
    /* Se usa el k/v estático global; ver multihead_loss_ctx. */
    extern tc_tensor_nd *g_mh_k, *g_mh_v;
    tc_tensor_nd *out =
        tc_tensor_nd_multihead_attention(q, g_mh_k, g_mh_v, 2, 0);
    tc_tensor_nd *loss = out ? tc_tensor_nd_sum(out) : NULL;
    const float value = loss ? tc_tensor_nd_const_data(loss)[0] : NAN;
    tc_tensor_nd_free(loss);
    tc_tensor_nd_free(out);
    return value;
}

tc_tensor_nd *g_mh_k = NULL;
tc_tensor_nd *g_mh_v = NULL;

/* Bloque estático para las pruebas de gradiente. */
static tc_transformer_block *g_block = NULL;
static tc_tensor_nd *g_block_x = NULL;

static float block_input_loss(tc_tensor_nd *x) {
    tc_tensor_nd *out = tc_transformer_block_forward(g_block, x);
    tc_tensor_nd *loss = out ? tc_tensor_nd_sum(out) : NULL;
    const float value = loss ? tc_tensor_nd_const_data(loss)[0] : NAN;
    tc_tensor_nd_free(loss);
    tc_tensor_nd_free(out);
    return value;
}

static float block_weight_loss(tc_tensor_nd *w) {
    (void)w; /* el peso se perturba en su lugar; el bloque lo referencia */
    tc_tensor_nd *out = tc_transformer_block_forward(g_block, g_block_x);
    tc_tensor_nd *loss = out ? tc_tensor_nd_sum(out) : NULL;
    const float value = loss ? tc_tensor_nd_const_data(loss)[0] : NAN;
    tc_tensor_nd_free(loss);
    tc_tensor_nd_free(out);
    return value;
}

/* --- permute: valores y gradiente --- */
static int test_permute(void) {
    const size_t shape[] = {2, 3, 4};
    float values[24];
    for (size_t i = 0; i < 24; ++i) values[i] = (float)(i + 1) * 0.5f;
    tc_tensor_nd *x = tc_tensor_nd_from_data(shape, 3, values, 1);
    const size_t axes[] = {2, 0, 1};
    tc_tensor_nd *p = tc_tensor_nd_permute(x, axes, 3);
    int ok = x && p && tc_tensor_nd_ndim(p) == 3 &&
             tc_tensor_nd_shape(p)[0] == 4 &&
             tc_tensor_nd_shape(p)[1] == 2 &&
             tc_tensor_nd_shape(p)[2] == 3;
    /* Referencia: out[a,b,c] = x[c,a,b] (x2d row-major [2,3,4]). */
    const float *xd = tc_tensor_nd_const_data(x);
    const float *pd = tc_tensor_nd_const_data(p);
    for (size_t a = 0; ok && a < 4; ++a)
        for (size_t b = 0; ok && b < 2; ++b)
            for (size_t c = 0; ok && c < 3; ++c)
                ok = close_enough(pd[(a * 2 + b) * 3 + c],
                                  xd[(b * 3 + c) * 4 + a]);
    /* Ejes inválidos deben rechazarse. */
    {
        const size_t bad1[] = {0, 0, 1};
        const size_t bad2[] = {0, 1, 3};
        ok = ok && !tc_tensor_nd_permute(x, bad1, 3) &&
             !tc_tensor_nd_permute(x, bad2, 3) &&
             !tc_tensor_nd_permute(x, axes, 2);
    }
    if (ok) {
        /* w con la forma PERMUTADA [4,2,3]: out[a,b,c] = x[c,a,b]. */
        const size_t wshape[] = {4, 2, 3};
        float wv[24];
        for (size_t i = 0; i < 24; ++i)
            wv[i] = 0.05f + 0.01f * (float)(i % 7);
        g_perm_w = tc_tensor_nd_from_data(wshape, 3, wv, 0);
        tc_tensor_nd *axes3 = tc_tensor_nd_permute(x, axes, 3);
        tc_tensor_nd *m = axes3 ? tc_tensor_nd_mul(axes3, g_perm_w) : NULL;
        tc_tensor_nd *loss = m ? tc_tensor_nd_sum(m) : NULL;
        ok = loss && tc_tensor_nd_backward(loss);
        tc_tensor_nd_free(loss);
        tc_tensor_nd_free(m);
        tc_tensor_nd_free(axes3);
        ok = ok && g_perm_w &&
             finite_difference(x, permute_loss, tc_tensor_nd_grad(x));
        tc_tensor_nd_free(g_perm_w);
        g_perm_w = NULL;
    }
    tc_tensor_nd_free(p);
    tc_tensor_nd_free(x);
    return ok;
}

/* --- posiciones sinusoidales: valores conocidos --- */
static int test_sinusoidal(void) {
    tc_tensor_nd *pe = tc_tensor_nd_sinusoidal_positions(4, 6);
    int ok = pe && tc_tensor_nd_shape(pe)[0] == 4 &&
             tc_tensor_nd_shape(pe)[1] == 6;
    if (ok) {
        const float *d = tc_tensor_nd_const_data(pe);
        /* pos 0: sin(0)=0 en pares, cos(0)=1 en impares. */
        ok = close_enough(d[0], 0.0f) && close_enough(d[1], 1.0f) &&
             close_enough(d[2], 0.0f) && close_enough(d[3], 1.0f);
        /* pos 1, i=0: sin(1) = 0.8414...; i=1: cos(1) = 0.5403... */
        ok = ok && close_enough(d[6], sinf(1.0f)) &&
             close_enough(d[7], cosf(1.0f));
    }
    ok = ok && !tc_tensor_nd_sinusoidal_positions(0, 4) &&
         !tc_tensor_nd_sinusoidal_positions(4, 0);
    tc_tensor_nd_free(pe);
    return ok;
}

/* --- multi-head: valores contra referencia independiente y gradiente --- */
static int test_multihead(void) {
    const size_t b = 2, s = 3, d = 4, heads = 2, hd = d / heads;
    const size_t shape[] = {b, s, d};
    float qv[24], kv[24], vv[24];
    for (size_t i = 0; i < 24; ++i) {
        /* Ojo: aritmética con signo antes de restar (size_t underflowa). */
        qv[i] = 0.3f * (float)((int)((i * 7) % 11) - 5);
        kv[i] = 0.2f * (float)((int)((i * 5) % 13) - 6);
        vv[i] = 0.4f * (float)((int)((i * 3) % 7) - 3);
    }
    tc_tensor_nd *q = tc_tensor_nd_from_data(shape, 3, qv, 1);
    tc_tensor_nd *k = tc_tensor_nd_from_data(shape, 3, kv, 0);
    tc_tensor_nd *v = tc_tensor_nd_from_data(shape, 3, vv, 0);
    tc_tensor_nd *out = tc_tensor_nd_multihead_attention(q, k, v, heads, 0);
    int ok = q && k && v && out && tc_tensor_nd_ndim(out) == 3 &&
             tc_tensor_nd_shape(out)[0] == b &&
             tc_tensor_nd_shape(out)[1] == s &&
             tc_tensor_nd_shape(out)[2] == d;

    /* Referencia: atención por cabeza con bucles directos. */
    const float *qd = tc_tensor_nd_const_data(q);
    const float *kd = tc_tensor_nd_const_data(k);
    const float *vd = tc_tensor_nd_const_data(v);
    const float *od = tc_tensor_nd_const_data(out);
    const float scale = 1.0f / sqrtf((float)hd);
    for (size_t n = 0; ok && n < b; ++n)
        for (size_t h = 0; ok && h < heads; ++h)
            for (size_t i = 0; ok && i < s; ++i) {
                float probs[3];
                float sum = 0.0f;
                for (size_t j = 0; j < s; ++j) {
                    float score = 0.0f;
                    for (size_t e = 0; e < hd; ++e) {
                        /* split_heads: qh[n,h,i,e] = q[n,i,h*hd+e]. */
                        const size_t qi = (n * s + i) * d + h * hd + e;
                        const size_t ki = (n * s + j) * d + h * hd + e;
                        score += qd[qi] * kd[ki];
                    }
                    probs[j] = expf(score * scale);
                    sum += probs[j];
                }
                for (size_t e = 0; ok && e < hd; ++e) {
                    float expected = 0.0f;
                    for (size_t j = 0; j < s; ++j) {
                        const size_t vi = (n * s + j) * d + h * hd + e;
                        expected += (probs[j] / sum) * vd[vi];
                    }
                    /* out[n, i, h*hd+e] */
                    ok = close_enough(od[(n * s + i) * d + h * hd + e],
                                      expected);
                }
            }

    /* Gradientes por diferencias finitas respecto a q. */
    if (ok) {
        g_mh_k = k;
        g_mh_v = v;
        tc_tensor_nd *loss = tc_tensor_nd_sum(out);
        ok = loss && tc_tensor_nd_backward(loss);
        tc_tensor_nd_free(loss);
        ok = ok && finite_difference(q, multihead_loss, tc_tensor_nd_grad(q));
        g_mh_k = NULL;
        g_mh_v = NULL;
    }

    /* Rechazos: heads inválidos y formas incompatibles. */
    {
        const size_t bad_shape[] = {b, s, 5};
        float badd[30] = {0};
        tc_tensor_nd *bad = tc_tensor_nd_from_data(bad_shape, 3, badd, 0);
        ok = ok && !tc_tensor_nd_multihead_attention(q, k, v, 3, 0) &&
             !tc_tensor_nd_multihead_attention(q, k, v, 0, 0) &&
             !tc_tensor_nd_multihead_attention(q, bad, v, 2, 0);
        tc_tensor_nd_free(bad);
    }

    tc_tensor_nd_free(out);
    tc_tensor_nd_free(q);
    tc_tensor_nd_free(k);
    tc_tensor_nd_free(v);
    return ok;
}

/* --- bloque transformer: formas, determinismo, gradientes --- */
static int test_block(void) {
    g_block = tc_transformer_block_create(4, 8, 2, 42);
    const size_t xshape[] = {2, 3, 4};
    float xv[24];
    for (size_t i = 0; i < 24; ++i)
        xv[i] = 0.25f * (float)((int)((i * 5) % 9) - 4);
    g_block_x = tc_tensor_nd_from_data(xshape, 3, xv, 1);
    tc_tensor_nd *out = tc_transformer_block_forward(g_block, g_block_x);
    int ok = g_block && g_block_x && out &&
             tc_tensor_nd_ndim(out) == 3 &&
             tc_tensor_nd_shape(out)[0] == 2 &&
             tc_tensor_nd_shape(out)[1] == 3 &&
             tc_tensor_nd_shape(out)[2] == 4;
    ok = ok && tc_transformer_block_param_count(g_block) == 8 &&
         tc_transformer_block_param(g_block, 7) &&
         !tc_transformer_block_param(g_block, 8);

    /* Determinismo: mismo seed, mismos pesos y salida. */
    if (ok) {
        tc_transformer_block *other = tc_transformer_block_create(4, 8, 2, 42);
        tc_tensor_nd *out2 = tc_transformer_block_forward(other, g_block_x);
        const float *a = tc_tensor_nd_const_data(out);
        const float *b = tc_tensor_nd_const_data(out2);
        for (size_t i = 0; ok && i < 24; ++i) ok = close_enough(a[i], b[i]);
        tc_tensor_nd_free(out2);
        tc_transformer_block_free(other);
    }

    /* Gradiente analítico vs diferencias finitas (entrada y W_q). */
    if (ok) {
        tc_tensor_nd *loss = tc_tensor_nd_sum(out);
        ok = loss && tc_tensor_nd_backward(loss);
        tc_tensor_nd_free(loss);
        ok = ok && finite_difference(g_block_x, block_input_loss,
                                     tc_tensor_nd_grad(g_block_x));
        /* Los gradientes se acumularon también en los pesos; limpiar antes
         * de medir W_q por separado. */
        for (size_t i = 0; i < 8; ++i)
            tc_tensor_nd_zero_grad(tc_transformer_block_param(g_block, i));
        tc_tensor_nd_zero_grad(g_block_x);
        loss = NULL;
        {
            tc_tensor_nd *fresh = tc_transformer_block_forward(g_block,
                                                               g_block_x);
            loss = fresh ? tc_tensor_nd_sum(fresh) : NULL;
            ok = ok && loss && tc_tensor_nd_backward(loss);
            tc_tensor_nd_free(loss);
            tc_tensor_nd_free(fresh);
        }
        ok = ok && finite_difference(tc_transformer_block_param(g_block, 0),
                                     block_weight_loss,
                                     tc_tensor_nd_grad(
                                         tc_transformer_block_param(g_block,
                                                                    0)));
    }

    /* Rechazos. */
    {
        const size_t bad_shape[] = {2, 3, 5};
        float badd[30] = {0};
        tc_tensor_nd *bad = tc_tensor_nd_from_data(bad_shape, 3, badd, 0);
        ok = ok && !tc_transformer_block_create(4, 8, 3, 1) &&
             !tc_transformer_block_create(0, 8, 2, 1) &&
             !tc_transformer_block_forward(g_block, bad);
        tc_tensor_nd_free(bad);
    }

    tc_tensor_nd_free(out);
    tc_tensor_nd_free(g_block_x);
    g_block_x = NULL;
    tc_transformer_block_free(g_block);
    g_block = NULL;
    return ok;
}

/* --- entrenamiento: el bloque aprende el promedio de la secuencia --- */
static int test_block_training(void) {
    const size_t b = 2, s = 3, d = 4;
    const size_t xshape[] = {b, s, d};
    const size_t yshape[] = {b, s, d};
    float xs[24], ys[24];
    unsigned int st = 9;
    for (size_t i = 0; i < 24; ++i) {
        /* Valores fijos deterministas en [-1, 1]. */
        st = st * 1664525u + 1013904223u;
        xs[i] = ((float)(st >> 8) / 16777216.0f) * 2.0f - 1.0f;
        ys[i] = 0.0f;
    }
    /* Objetivo: en cada posición, el promedio de la secuencia. */
    for (size_t n = 0; n < b; ++n)
        for (size_t e = 0; e < d; ++e) {
            float mean = 0.0f;
            for (size_t t = 0; t < s; ++t) mean += xs[(n * s + t) * d + e];
            mean /= (float)s;
            for (size_t t = 0; t < s; ++t) ys[(n * s + t) * d + e] = mean;
        }

    tc_transformer_block *block = tc_transformer_block_create(d, 8, 2, 5);
    tc_tensor_nd *x = tc_tensor_nd_from_data(xshape, 3, xs, 0);
    tc_tensor_nd *y = tc_tensor_nd_from_data(yshape, 3, ys, 0);
    tc_tensor_nd *params[8];
    for (size_t i = 0; i < 8; ++i)
        params[i] = tc_transformer_block_param(block, i);
    tc_tensor_nd_optimizer *adam = tc_tensor_nd_optimizer_adam(params, 8, 0.01f);

    float first_loss = 0.0f, last_loss = 0.0f;
    int ok = block && x && y && adam;
    for (int step = 0; ok && step < 300; ++step) {
        tc_tensor_nd *pred = tc_transformer_block_forward(block, x);
        tc_tensor_nd *loss = pred ? tc_tensor_nd_mse_loss(pred, y) : NULL;
        ok = ok && loss && tc_tensor_nd_backward(loss);
        if (step == 0) first_loss = tc_tensor_nd_const_data(loss)[0];
        last_loss = tc_tensor_nd_const_data(loss)[0];
        tc_tensor_nd_optimizer_step(adam);
        tc_tensor_nd_free(loss);
        tc_tensor_nd_free(pred);
    }
    ok = ok && last_loss < first_loss * 0.5f;
    if (!ok && block)
        fprintf(stderr, "  entrenamiento: first=%f last=%f\n", first_loss,
                last_loss);

    tc_tensor_nd_optimizer_free(adam);
    tc_tensor_nd_free(x);
    tc_tensor_nd_free(y);
    tc_transformer_block_free(block);
    return ok;
}

int main(void) {
    const int permute_ok = test_permute();
    const int sinusoidal_ok = test_sinusoidal();
    const int multihead_ok = test_multihead();
    const int block_ok = test_block();
    const int training_ok = test_block_training();

    printf("permute: %s\n", permute_ok ? "ok" : "error");
    printf("sinusoidal: %s\n", sinusoidal_ok ? "ok" : "error");
    printf("multihead: %s\n", multihead_ok ? "ok" : "error");
    printf("block: %s\n", block_ok ? "ok" : "error");
    printf("block training: %s\n", training_ok ? "ok" : "error");

    return (permute_ok && sinusoidal_ok && multihead_ok && block_ok &&
            training_ok) ? 0 : 1;
}
