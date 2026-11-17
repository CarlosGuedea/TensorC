#include <tensorc/tensorc.h>

#include <math.h>
#include <stdio.h>

static int close_enough(float actual, float expected) {
    return fabsf(actual - expected) < 1e-4f;
}

static int gradient_close(float analytic, float numeric) {
    return fabsf(analytic - numeric) <= 2e-3f * (1.0f + fabsf(numeric));
}

static int test_softmax(void) {
    const size_t shape[] = {1, 4};
    const float values[] = {1.0f, 2.0f, 3.0f, 4.0f};
    tc_tensor_nd *x = tc_tensor_nd_from_data(shape, 2, values, 1);
    tc_tensor_nd *probs = tc_tensor_nd_softmax(x);
    int ok = x && probs;
    const float *p = tc_tensor_nd_const_data(probs);
    /* Valores de referencia: exp(x)/sum(exp(x)). */
    const float expected[] = {0.0320585f, 0.0871443f, 0.2368830f, 0.6439143f};
    float total = 0.0f;
    for (size_t i = 0; ok && i < 4; ++i) {
        ok = fabsf(p[i] - expected[i]) < 1e-5f;
        total += p[i];
    }
    ok = ok && fabsf(total - 1.0f) < 1e-5f;

    /* Gradiente por diferencias finitas de sum(softmax(x) * c). */
    const size_t c_shape[] = {1, 4};
    const float c_values[] = {0.5f, -1.0f, 2.0f, 0.25f};
    tc_tensor_nd *c = tc_tensor_nd_from_data(c_shape, 2, c_values, 0);
    tc_tensor_nd *weighted = tc_tensor_nd_mul(probs, c);
    tc_tensor_nd *loss = tc_tensor_nd_sum(weighted);
    ok = ok && c && weighted && loss && tc_tensor_nd_backward(loss);
    const float *grad = tc_tensor_nd_grad(x);
    if (ok) {
        const float epsilon = 1e-3f;
        float *data = tc_tensor_nd_data(x);
        for (size_t i = 0; ok && i < 4; ++i) {
            const float original = data[i];
            data[i] = original + epsilon;
            tc_tensor_nd *plus_out = tc_tensor_nd_softmax(x);
            tc_tensor_nd *plus_w = tc_tensor_nd_mul(plus_out, c);
            tc_tensor_nd *plus_l = tc_tensor_nd_sum(plus_w);
            const float plus = tc_tensor_nd_const_data(plus_l)[0];
            data[i] = original - epsilon;
            tc_tensor_nd *minus_out = tc_tensor_nd_softmax(x);
            tc_tensor_nd *minus_w = tc_tensor_nd_mul(minus_out, c);
            tc_tensor_nd *minus_l = tc_tensor_nd_sum(minus_w);
            const float minus = tc_tensor_nd_const_data(minus_l)[0];
            data[i] = original;
            const float numeric = (plus - minus) / (2.0f * epsilon);
            ok = gradient_close(grad[i], numeric);
            tc_tensor_nd_free(minus_l);
            tc_tensor_nd_free(minus_w);
            tc_tensor_nd_free(minus_out);
            tc_tensor_nd_free(plus_l);
            tc_tensor_nd_free(plus_w);
            tc_tensor_nd_free(plus_out);
        }
    }

    tc_tensor_nd_free(loss);
    tc_tensor_nd_free(weighted);
    tc_tensor_nd_free(c);
    tc_tensor_nd_free(probs);
    tc_tensor_nd_free(x);
    return ok;
}

static int test_cross_entropy(void) {
    const size_t shape[] = {2, 3};
    const float probs[] = {0.7f, 0.2f, 0.1f, 0.1f, 0.8f, 0.1f};
    const float targets[] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    tc_tensor_nd *p = tc_tensor_nd_from_data(shape, 2, probs, 1);
    tc_tensor_nd *t = tc_tensor_nd_from_data(shape, 2, targets, 0);
    tc_tensor_nd *loss = tc_tensor_nd_cross_entropy(p, t);
    /* (-ln 0.7 - ln 0.8) / 2. */
    int ok = p && t && loss &&
             close_enough(tc_tensor_nd_const_data(loss)[0], 0.2899093f);

    /* Gradiente de la composición softmax + CE = (p - t) / filas. */
    const size_t logit_shape[] = {2, 3};
    const float logits[] = {1.5f, -0.5f, 0.25f, -1.0f, 2.0f, 0.5f};
    tc_tensor_nd *x = tc_tensor_nd_from_data(logit_shape, 2, logits, 1);
    tc_tensor_nd *sm = tc_tensor_nd_softmax(x);
    tc_tensor_nd *ce = tc_tensor_nd_cross_entropy(sm, t);
    ok = ok && x && sm && ce && tc_tensor_nd_backward(ce);
    if (ok) {
        const float *prob_data = tc_tensor_nd_const_data(sm);
        const float *grad = tc_tensor_nd_grad(x);
        for (size_t i = 0; ok && i < 6; ++i)
            ok = gradient_close(grad[i], (prob_data[i] - targets[i]) / 2.0f);
    }

    /* Rechazo de formas incompatibles. */
    const size_t other_shape[] = {3, 2};
    tc_tensor_nd *bad = tc_tensor_nd_from_data(other_shape, 2, probs, 0);
    tc_tensor_nd *invalid = tc_tensor_nd_cross_entropy(p, bad);
    ok = ok && bad && !invalid;

    tc_tensor_nd_free(invalid);
    tc_tensor_nd_free(bad);
    tc_tensor_nd_free(ce);
    tc_tensor_nd_free(sm);
    tc_tensor_nd_free(x);
    tc_tensor_nd_free(loss);
    tc_tensor_nd_free(t);
    tc_tensor_nd_free(p);
    return ok;
}

/* Adam minimiza mean((w - 3)^2) con mse_loss: w debe acercarse a 3. */
static int test_adam(void) {
    const size_t shape[] = {4};
    const float initial[] = {0.0f, -2.0f, 5.0f, 1.0f};
    const float three[] = {3.0f, 3.0f, 3.0f, 3.0f};
    tc_tensor_nd *w = tc_tensor_nd_from_data(shape, 1, initial, 1);
    tc_tensor_nd *target = tc_tensor_nd_from_data(shape, 1, three, 0);
    int ok = w && target;
    tc_tensor_nd *params[] = {w};
    tc_tensor_nd_optimizer *optimizer =
        ok ? tc_tensor_nd_optimizer_adam(params, 1, 0.03f) : NULL;
    ok = ok && optimizer;
    for (int step = 0; ok && step < 400; ++step) {
        tc_tensor_nd *loss = tc_tensor_nd_mse_loss(w, target);
        ok = ok && loss && tc_tensor_nd_backward(loss);
        if (ok && tc_tensor_nd_optimizer_step(optimizer) != 1) ok = 0;
        tc_tensor_nd_free(loss);
    }
    if (ok) {
        const float *data = tc_tensor_nd_const_data(w);
        for (size_t i = 0; ok && i < 4; ++i)
            ok = fabsf(data[i] - 3.0f) < 0.05f;
    }

    /* Un parámetro sin gradiente no se actualiza pero no rompe el paso. */
    tc_tensor_nd *frozen = tc_tensor_nd_create(shape, 1, 0);
    tc_tensor_nd *mixed[] = {w, frozen};
    tc_tensor_nd_optimizer *opt2 =
        ok ? tc_tensor_nd_optimizer_adam(mixed, 2, 0.2f) : NULL;
    ok = ok && frozen && opt2 && tc_tensor_nd_optimizer_step(opt2) == 1;

    tc_tensor_nd_optimizer_free(opt2);
    tc_tensor_nd_free(frozen);
    tc_tensor_nd_optimizer_free(optimizer);
    tc_tensor_nd_free(target);
    tc_tensor_nd_free(w);
    return ok;
}

int main(void) {
    const int softmax_ok = test_softmax();
    const int cross_entropy_ok = test_cross_entropy();
    const int adam_ok = test_adam();
    printf("softmax: %s\n", softmax_ok ? "ok" : "error");
    printf("cross_entropy: %s\n", cross_entropy_ok ? "ok" : "error");
    printf("adam: %s\n", adam_ok ? "ok" : "error");
    return (softmax_ok && cross_entropy_ok && adam_ok) ? 0 : 1;
}
