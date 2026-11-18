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

/* Contexto global para las funciones de pérdida (patrón de la suite). */
static tc_tensor_nd *g_ids = NULL;

/* loss = sum(embedding(table, ids)): ejercita el scatter-add del backward
 * (los índices repetidos deben acumular gradiente). */
static float embedding_table_loss(tc_tensor_nd *table) {
    tc_tensor_nd *out = tc_tensor_nd_embedding(table, g_ids);
    tc_tensor_nd *loss = out ? tc_tensor_nd_sum(out) : NULL;
    const float value = loss ? tc_tensor_nd_const_data(loss)[0] : NAN;
    tc_tensor_nd_free(loss);
    tc_tensor_nd_free(out);
    return value;
}

/* loss = cross_entropy_sparse(prediction, ids) directamente. */
static float sparse_ce_loss_fn(tc_tensor_nd *p) {
    tc_tensor_nd *loss = tc_tensor_nd_cross_entropy_sparse(p, g_ids);
    const float value = loss ? tc_tensor_nd_const_data(loss)[0] : NAN;
    tc_tensor_nd_free(loss);
    return value;
}

/* --- embedding: valores, gradiente y rechazos --- */
static int test_embedding(void) {
    const size_t vocab = 4, dim = 3;
    const size_t tshape[] = {vocab, dim};
    float tv[12];
    for (size_t i = 0; i < 12; ++i) tv[i] = 0.25f * (float)(i + 1);
    tc_tensor_nd *table = tc_tensor_nd_from_data(tshape, 2, tv, 1);
    /* ids [2, 3] con repetición: la salida es [2, 3, dim]. */
    const size_t ishape[] = {2, 3};
    const float iv[] = {0, 2, 2, 3, 1, 0};
    tc_tensor_nd *ids = tc_tensor_nd_from_data(ishape, 2, iv, 0);
    tc_tensor_nd *out = tc_tensor_nd_embedding(table, ids);
    int ok = table && ids && out && tc_tensor_nd_ndim(out) == 3 &&
             tc_tensor_nd_shape(out)[0] == 2 &&
             tc_tensor_nd_shape(out)[1] == 3 &&
             tc_tensor_nd_shape(out)[2] == dim;
    const float *td = tc_tensor_nd_const_data(table);
    const float *od = tc_tensor_nd_const_data(out);
    for (size_t i = 0; ok && i < 6; ++i) {
        const size_t row = (size_t)iv[i];
        for (size_t d = 0; ok && d < dim; ++d)
            ok = close_enough(od[i * dim + d], td[row * dim + d]);
    }
    /* Gradiente analítico de la tabla vs diferencias finitas. */
    if (ok) {
        g_ids = ids;
        tc_tensor_nd *loss = tc_tensor_nd_sum(out);
        ok = loss && tc_tensor_nd_backward(loss);
        tc_tensor_nd_free(loss);
        /* Referencia directa: cada fila suma tantos unos como veces aparece
         * su índice en ids (loss = sum del gather). */
        const float *grad = tc_tensor_nd_grad(table);
        const float counts[] = {2, 1, 2, 1};
        for (size_t r = 0; ok && r < vocab; ++r)
            for (size_t d = 0; ok && d < dim; ++d)
                ok = close_enough(grad[r * dim + d], counts[r]);
        ok = ok && finite_difference(table, embedding_table_loss, grad);
        g_ids = NULL;
    }
    /* Rechazos: ids fuera de rango, no enteros y formas inválidas. */
    {
        const float bad_range[] = {0, 4};
        const float bad_frac[] = {1.5f, 0};
        const float bad_neg[] = {-1, 0};
        const size_t bshape[] = {2};
        tc_tensor_nd *b1 = tc_tensor_nd_from_data(bshape, 1, bad_range, 0);
        tc_tensor_nd *b2 = tc_tensor_nd_from_data(bshape, 1, bad_frac, 0);
        tc_tensor_nd *b3 = tc_tensor_nd_from_data(bshape, 1, bad_neg, 0);
        ok = ok && b1 && b2 && b3 && !tc_tensor_nd_embedding(table, b1) &&
             !tc_tensor_nd_embedding(table, b2) &&
             !tc_tensor_nd_embedding(table, b3) &&
             !tc_tensor_nd_embedding(ids, ids);
        tc_tensor_nd_free(b1);
        tc_tensor_nd_free(b2);
        tc_tensor_nd_free(b3);
    }
    tc_tensor_nd_free(out);
    tc_tensor_nd_free(ids);
    tc_tensor_nd_free(table);
    return ok;
}

/* --- cross-entropy sparse: valor, equivalencia con one-hot, gradiente --- */
static int test_sparse_ce(void) {
    const size_t rows = 2, classes = 3;
    const size_t pshape[] = {rows, classes};
    const float pv[] = {0.2f, 0.3f, 0.5f,
                        0.6f, 0.1f, 0.3f};
    tc_tensor_nd *p = tc_tensor_nd_from_data(pshape, 2, pv, 1);
    const size_t ishape[] = {rows};
    const float iv[] = {2, 0};
    tc_tensor_nd *ids = tc_tensor_nd_from_data(ishape, 1, iv, 0);
    tc_tensor_nd *loss = tc_tensor_nd_cross_entropy_sparse(p, ids);
    int ok = p && ids && loss && tc_tensor_nd_numel(loss) == 1;

    if (ok) {
        const float expected =
            -(logf(0.5f) + logf(0.6f)) / 2.0f;
        ok = close_enough(tc_tensor_nd_const_data(loss)[0], expected);
    }
    /* Equivalencia con la CE densa sobre el one-hot correspondiente. */
    if (ok) {
        const float onehot[] = {0, 0, 1,
                                1, 0, 0};
        tc_tensor_nd *t = tc_tensor_nd_from_data(pshape, 2, onehot, 0);
        tc_tensor_nd *dense = tc_tensor_nd_cross_entropy(p, t);
        ok = t && dense &&
             close_enough(tc_tensor_nd_const_data(loss)[0],
                          tc_tensor_nd_const_data(dense)[0]);
        tc_tensor_nd_free(dense);
        tc_tensor_nd_free(t);
    }
    /* Gradiente analítico vs diferencias finitas sobre prediction. */
    if (ok) {
        g_ids = ids;
        ok = tc_tensor_nd_backward(loss);
        const float *grad = tc_tensor_nd_grad(p);
        /* Referencia: solo p[0,2] y p[1,0] reciben -1/(rows*p). */
        ok = ok && close_enough(grad[2], -1.0f / (2.0f * 0.5f)) &&
             close_enough(grad[3], -1.0f / (2.0f * 0.6f)) &&
             close_enough(grad[0], 0.0f) && close_enough(grad[5], 0.0f);
        ok = ok && finite_difference(p, sparse_ce_loss_fn, grad);
        g_ids = NULL;
    }
    /* Rechazos: ids con longitud distinta de rows y fuera de rango. */
    {
        const size_t bad_shape[] = {3};
        const float bad_len[] = {0, 1, 2};
        const float bad_range[] = {3, 0};
        tc_tensor_nd *b1 = tc_tensor_nd_from_data(bad_shape, 1, bad_len, 0);
        tc_tensor_nd *b2 = tc_tensor_nd_from_data(ishape, 1, bad_range, 0);
        ok = ok && b1 && b2 &&
             !tc_tensor_nd_cross_entropy_sparse(p, b1) &&
             !tc_tensor_nd_cross_entropy_sparse(p, b2);
        tc_tensor_nd_free(b1);
        tc_tensor_nd_free(b2);
    }
    tc_tensor_nd_free(loss);
    tc_tensor_nd_free(ids);
    tc_tensor_nd_free(p);
    return ok;
}

/* --- set_lr: el mismo paso de Adam escala con el lr vigente --- */
static int test_optimizer_set_lr(void) {
    const size_t shape[] = {1};
    const float one[] = {1.0f};
    tc_tensor_nd *p1 = tc_tensor_nd_from_data(shape, 1, one, 1);
    tc_tensor_nd *p2 = tc_tensor_nd_from_data(shape, 1, one, 1);
    tc_tensor_nd *params1[] = {p1};
    tc_tensor_nd *params2[] = {p2};
    tc_tensor_nd_optimizer *a1 =
        tc_tensor_nd_optimizer_adam(params1, 1, 0.1f);
    tc_tensor_nd_optimizer *a2 =
        tc_tensor_nd_optimizer_adam(params2, 1, 0.1f);
    int ok = p1 && p2 && a1 && a2;
    /* Gradiente constante 1 en ambos (loss = sum(p)): el primer paso de
     * Adam mueve ~lr (m_hat=1, v_hat=1). */
    if (ok) {
        tc_tensor_nd *l1 = tc_tensor_nd_sum(p1);
        tc_tensor_nd *l2 = tc_tensor_nd_sum(p2);
        ok = l1 && l2 && tc_tensor_nd_backward(l1) &&
             tc_tensor_nd_backward(l2);
        tc_tensor_nd_free(l1);
        tc_tensor_nd_free(l2);
    }
    if (ok) {
        tc_tensor_nd_optimizer_set_lr(a2, 0.01f);
        ok = tc_tensor_nd_optimizer_step(a1) == 1 &&
             tc_tensor_nd_optimizer_step(a2) == 1;
        const float d1 = tc_tensor_nd_const_data(p1)[0];
        const float d2 = tc_tensor_nd_const_data(p2)[0];
        /* Primer paso de Adam: p -= lr * 1/(1+eps) ≈ lr. */
        ok = ok && close_enough(d1, 1.0f - 0.1f) &&
             close_enough(d2, 1.0f - 0.01f);
        /* NULL no rompe. */
        tc_tensor_nd_optimizer_set_lr(NULL, 1.0f);
    }
    tc_tensor_nd_optimizer_free(a1);
    tc_tensor_nd_optimizer_free(a2);
    tc_tensor_nd_free(p1);
    tc_tensor_nd_free(p2);
    return ok;
}

int main(void) {
    const int embedding_ok = test_embedding();
    const int sparse_ce_ok = test_sparse_ce();
    const int set_lr_ok = test_optimizer_set_lr();

    printf("embedding: %s\n", embedding_ok ? "ok" : "error");
    printf("sparse cross entropy: %s\n", sparse_ce_ok ? "ok" : "error");
    printf("optimizer set_lr: %s\n", set_lr_ok ? "ok" : "error");

    return (embedding_ok && sparse_ce_ok && set_lr_ok) ? 0 : 1;
}
