#include <tensorc/tensorc.h>

#include <math.h>
#include <stdio.h>

static int close_enough(float actual, float expected) {
    return fabsf(actual - expected) < 1e-5f;
}

static float conv2d_loss_value(tc_tensor_nd *input, tc_tensor_nd *kernel,
                               tc_conv2d_options options) {
    tc_tensor_nd *output = tc_tensor_nd_conv2d_ex(input, kernel, options);
    tc_tensor_nd *loss = tc_tensor_nd_sum(output);
    const float value = loss ? tc_tensor_nd_const_data(loss)[0] : NAN;
    tc_tensor_nd_free(loss);
    tc_tensor_nd_free(output);
    return value;
}

/* Tolerancia relativa alrededor de 1e-3: el error de diferencias finitas en
 * float32 crece con la magnitud del gradiente y de la pérdida. */
static int gradient_close(float analytic, float numeric) {
    return fabsf(analytic - numeric) <= 2e-3f * (1.0f + fabsf(numeric));
}

static int finite_difference_check(tc_tensor_nd *input, tc_tensor_nd *kernel,
                                   tc_conv2d_options options) {
    /* Con float32, eps=1e-3 deja el ruido de cancelación (ulp de la pérdida
     * dividido entre eps) por encima de la tolerancia cuando el gradiente
     * real es casi cero; eps=1e-2 lo mantiene dominado. */
    const float epsilon = 1e-2f;
    float *input_data = tc_tensor_nd_data(input);
    float *kernel_data = tc_tensor_nd_data(kernel);
    const float *input_gradient = tc_tensor_nd_grad(input);
    const float *kernel_gradient = tc_tensor_nd_grad(kernel);

    for (size_t i = 0; i < tc_tensor_nd_numel(input); ++i) {
        const float original = input_data[i];
        input_data[i] = original + epsilon;
        const float plus = conv2d_loss_value(input, kernel, options);
        input_data[i] = original - epsilon;
        const float minus = conv2d_loss_value(input, kernel, options);
        input_data[i] = original;
        const float numeric = (plus - minus) / (2.0f * epsilon);
        if (!gradient_close(input_gradient[i], numeric)) return 0;
    }

    for (size_t i = 0; i < tc_tensor_nd_numel(kernel); ++i) {
        const float original = kernel_data[i];
        kernel_data[i] = original + epsilon;
        const float plus = conv2d_loss_value(input, kernel, options);
        kernel_data[i] = original - epsilon;
        const float minus = conv2d_loss_value(input, kernel, options);
        kernel_data[i] = original;
        const float numeric = (plus - minus) / (2.0f * epsilon);
        if (!gradient_close(kernel_gradient[i], numeric)) return 0;
    }
    return 1;
}

int main(void) {
    const size_t input_shape[] = {1, 1, 3, 3};
    const size_t kernel_shape[] = {1, 1, 2, 2};
    const float input_values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    const float kernel_values[] = {1, 0, 0, 1};
    const float expected_output[] = {6, 8, 12, 14};
    const float expected_input_grad[] = {1, 1, 0, 1, 2, 1, 0, 1, 1};
    const float expected_kernel_grad[] = {12, 16, 24, 28};

    tc_tensor_nd *input = tc_tensor_nd_from_data(input_shape, 4, input_values, 1);
    tc_tensor_nd *kernel = tc_tensor_nd_from_data(kernel_shape, 4, kernel_values, 1);
    tc_tensor_nd *output = tc_tensor_nd_conv2d(input, kernel);
    tc_tensor_nd *loss = tc_tensor_nd_sum(output);
    int ok = input && kernel && output && loss && tc_tensor_nd_numel(output) == 4 &&
             tc_tensor_nd_backward(loss);
    const float *output_data = tc_tensor_nd_const_data(output);
    const float *input_grad = tc_tensor_nd_grad(input);
    const float *kernel_grad = tc_tensor_nd_grad(kernel);
    for (size_t i = 0; ok && i < 4; ++i)
        ok = close_enough(output_data[i], expected_output[i]);
    for (size_t i = 0; ok && i < 9; ++i)
        ok = close_enough(input_grad[i], expected_input_grad[i]);
    for (size_t i = 0; ok && i < 4; ++i)
        ok = close_enough(kernel_grad[i], expected_kernel_grad[i]);

    ok = ok && finite_difference_check(input, kernel, TC_CONV2D_DEFAULT);

    const size_t incompatible_shape[] = {1, 2, 2, 2};
    const float incompatible_values[] = {1, 1, 1, 1, 1, 1, 1, 1};
    tc_tensor_nd *incompatible = tc_tensor_nd_from_data(
        incompatible_shape, 4, incompatible_values, 0);
    tc_tensor_nd *invalid_output = tc_tensor_nd_conv2d(input, incompatible);
    ok = ok && incompatible && !invalid_output;

    /* Stride 2 sobre el caso conocido: solo sobrevive la ventana (0,0). */
    const tc_conv2d_options stride_two = {2, 2, 0, 0};
    tc_tensor_nd *strided = tc_tensor_nd_conv2d_ex(input, kernel, stride_two);
    ok = ok && strided && tc_tensor_nd_numel(strided) == 1 &&
         close_enough(tc_tensor_nd_const_data(strided)[0], 6.0f);

    /* Padding 1 sobre [[1,2],[3,4]] con kernel identidad diagonal. */
    const size_t small_shape[] = {1, 1, 2, 2};
    const float small_input[] = {1, 2, 3, 4};
    const float small_kernel[] = {1, 0, 0, 1};
    const float padded_expected[] = {1, 2, 0, 3, 5, 2, 0, 3, 4};
    const tc_conv2d_options padded_options = {1, 1, 1, 1};
    tc_tensor_nd *small_in = tc_tensor_nd_from_data(small_shape, 4, small_input, 1);
    tc_tensor_nd *small_k = tc_tensor_nd_from_data(small_shape, 4, small_kernel, 1);
    tc_tensor_nd *padded_out = tc_tensor_nd_conv2d_ex(small_in, small_k, padded_options);
    ok = ok && small_in && small_k && padded_out &&
         tc_tensor_nd_ndim(padded_out) == 4 &&
         tc_tensor_nd_shape(padded_out)[2] == 3 &&
         tc_tensor_nd_shape(padded_out)[3] == 3;
    const float *padded_data = tc_tensor_nd_const_data(padded_out);
    for (size_t i = 0; ok && i < 9; ++i)
        ok = close_enough(padded_data[i], padded_expected[i]);

    /* Gradientes con stride y padding verificados por diferencias finitas. */
    tc_tensor_nd *padded_loss = tc_tensor_nd_sum(padded_out);
    ok = ok && padded_loss && tc_tensor_nd_backward(padded_loss);
    tc_tensor_nd_free(padded_loss);
    ok = ok && finite_difference_check(small_in, small_k, padded_options);
    tc_tensor_nd_free(padded_out);
    tc_tensor_nd_free(small_k);
    tc_tensor_nd_free(small_in);

    /*
     * Identidad de composición con varios canales y filtros:
     * conv2d_ex(x, k, s, p) debe coincidir con conv2d(pad(x), k) tomando
     * un elemento cada s. El caso stride 1 / sin padding ya está validado
     * arriba, así que esto verifica el camino nuevo de forma independiente.
     */
    const size_t multi_shape[] = {2, 3, 4, 4};
    const size_t multi_kernel_shape[] = {5, 3, 3, 3};
    float multi_values[2 * 3 * 4 * 4];
    float multi_kernel_values[5 * 3 * 3 * 3];
    for (size_t i = 0; i < sizeof(multi_values) / sizeof(multi_values[0]); ++i)
        multi_values[i] = (float)((int)(i % 17) - 8) * 0.25f;
    for (size_t i = 0; i < sizeof(multi_kernel_values) / sizeof(multi_kernel_values[0]); ++i)
        multi_kernel_values[i] = (float)((int)(i % 13) - 6) * 0.5f;
    const tc_conv2d_options multi_options = {2, 2, 1, 1};
    tc_tensor_nd *multi_in = tc_tensor_nd_from_data(
        multi_shape, 4, multi_values, 1);
    tc_tensor_nd *multi_k = tc_tensor_nd_from_data(
        multi_kernel_shape, 4, multi_kernel_values, 1);
    tc_tensor_nd *multi_out = tc_tensor_nd_conv2d_ex(multi_in, multi_k, multi_options);
    ok = ok && multi_in && multi_k && multi_out &&
         tc_tensor_nd_shape(multi_out)[0] == 2 &&
         tc_tensor_nd_shape(multi_out)[1] == 5 &&
         tc_tensor_nd_shape(multi_out)[2] == 2 &&
         tc_tensor_nd_shape(multi_out)[3] == 2;

    const size_t padded_shape[] = {2, 3, 6, 6};
    tc_tensor_nd *padded_in = tc_tensor_nd_create(padded_shape, 4, 0);
    if (padded_in) {
        float *dst = tc_tensor_nd_data(padded_in);
        const float *src = tc_tensor_nd_const_data(multi_in);
        for (size_t n = 0; n < 2; ++n)
            for (size_t c = 0; c < 3; ++c)
                for (size_t h = 0; h < 4; ++h)
                    for (size_t w = 0; w < 4; ++w)
                        dst[((n * 3 + c) * 6 + h + 1) * 6 + w + 1] =
                            src[((n * 3 + c) * 4 + h) * 4 + w];
    }
    tc_tensor_nd *reference = tc_tensor_nd_conv2d(padded_in, multi_k);
    ok = ok && padded_in && reference;
    if (ok) {
        const float *expected = tc_tensor_nd_const_data(reference);
        const float *actual = tc_tensor_nd_const_data(multi_out);
        for (size_t n = 0; ok && n < 2; ++n)
            for (size_t k = 0; k < 5; ++k)
                for (size_t oh = 0; oh < 2; ++oh)
                    for (size_t ow = 0; ow < 2; ++ow)
                        ok = close_enough(
                            actual[((n * 5 + k) * 2 + oh) * 2 + ow],
                            expected[((n * 5 + k) * 4 + oh * 2) * 4 + ow * 2]);
    }

    /* Gradientes del caso multicanal con stride/padding por diferencias finitas. */
    tc_tensor_nd *multi_loss = tc_tensor_nd_sum(multi_out);
    ok = ok && multi_loss && tc_tensor_nd_backward(multi_loss);
    tc_tensor_nd_free(multi_loss);
    ok = ok && finite_difference_check(multi_in, multi_k, multi_options);

    /* Rechazos nuevos: stride 0 y kernel mayor que la entrada incluso con padding. */
    const tc_conv2d_options zero_stride = {0, 1, 0, 0};
    const tc_conv2d_options no_padding = {1, 1, 0, 0};
    const size_t oversized_shape[] = {1, 1, 4, 4};
    const float oversized_values[16] = {0};
    tc_tensor_nd *bad_stride = tc_tensor_nd_conv2d_ex(input, kernel, zero_stride);
    tc_tensor_nd *too_big_kernel = tc_tensor_nd_from_data(
        oversized_shape, 4, oversized_values, 0);
    tc_tensor_nd *bad_output = too_big_kernel
        ? tc_tensor_nd_conv2d_ex(input, too_big_kernel, no_padding) : NULL;
    ok = ok && !bad_stride && too_big_kernel && !bad_output;

    /*
     * Caso grande (supera el umbral de la ruta im2col/GEMM cuando Accelerate
     * está disponible): forward contra la identidad de composición y
     * gradientes por diferencias finitas. Datos pequeños en magnitud para
     * que el ruido de float32 no domine el chequeo numérico.
     */
    const size_t big_shape[] = {2, 3, 24, 24};
    const size_t large_kernel_shape[] = {4, 3, 3, 3};
    float big_values[2 * 3 * 24 * 24];
    float large_kernel_values[4 * 3 * 3 * 3];
    for (size_t i = 0; i < sizeof(big_values) / sizeof(big_values[0]); ++i)
        big_values[i] = (float)((int)(i % 17) - 8) * 5e-3f;
    for (size_t i = 0; i < sizeof(large_kernel_values) / sizeof(large_kernel_values[0]); ++i)
        large_kernel_values[i] = (float)((int)(i % 13) - 6) * 5e-3f;
    const tc_conv2d_options big_options = {1, 1, 1, 1};
    tc_tensor_nd *big_in = tc_tensor_nd_from_data(big_shape, 4, big_values, 1);
    tc_tensor_nd *big_k = tc_tensor_nd_from_data(
        large_kernel_shape, 4, large_kernel_values, 1);
    tc_tensor_nd *big_out = tc_tensor_nd_conv2d_ex(big_in, big_k, big_options);
    ok = ok && big_in && big_k && big_out;

    const size_t big_padded_shape[] = {2, 3, 26, 26};
    tc_tensor_nd *big_padded = tc_tensor_nd_create(big_padded_shape, 4, 0);
    if (big_padded) {
        float *dst = tc_tensor_nd_data(big_padded);
        for (size_t n = 0; n < 2; ++n)
            for (size_t c = 0; c < 3; ++c)
                for (size_t h = 0; h < 24; ++h)
                    for (size_t w = 0; w < 24; ++w)
                        dst[((n * 3 + c) * 26 + h + 1) * 26 + w + 1] =
                            big_values[((n * 3 + c) * 24 + h) * 24 + w];
    }
    tc_tensor_nd *big_reference = tc_tensor_nd_conv2d(big_padded, big_k);
    ok = ok && big_padded && big_reference &&
         tc_tensor_nd_numel(big_out) == tc_tensor_nd_numel(big_reference);
    if (ok) {
        const float *expected = tc_tensor_nd_const_data(big_reference);
        const float *actual = tc_tensor_nd_const_data(big_out);
        for (size_t i = 0; ok && i < tc_tensor_nd_numel(big_out); ++i)
            ok = close_enough(actual[i], expected[i]);
    }

    /* Mismo caso con stride 2: debe ser el submuestreo de la referencia. */
    const tc_conv2d_options big_stride2 = {2, 2, 1, 1};
    tc_tensor_nd *big_out2 = tc_tensor_nd_conv2d_ex(big_in, big_k, big_stride2);
    ok = ok && big_out2 && tc_tensor_nd_shape(big_out2)[2] == 12 &&
         tc_tensor_nd_shape(big_out2)[3] == 12;
    if (ok) {
        const float *expected = tc_tensor_nd_const_data(big_reference);
        const float *actual = tc_tensor_nd_const_data(big_out2);
        for (size_t n = 0; ok && n < 2; ++n)
            for (size_t k2 = 0; k2 < 4 && ok; ++k2)
                for (size_t oh = 0; oh < 12 && ok; ++oh)
                    for (size_t ow = 0; ow < 12 && ok; ++ow)
                        ok = close_enough(
                            actual[((n * 4 + k2) * 12 + oh) * 12 + ow],
                            expected[((n * 4 + k2) * 24 + oh * 2) * 24 + ow * 2]);
    }

    tc_tensor_nd *big_loss = tc_tensor_nd_sum(big_out);
    ok = ok && big_loss && tc_tensor_nd_backward(big_loss);
    tc_tensor_nd_free(big_loss);
    ok = ok && finite_difference_check(big_in, big_k, big_options);

    printf("conv2d: %s\n", ok ? "ok" : "error");
    tc_tensor_nd_free(bad_output);
    tc_tensor_nd_free(too_big_kernel);
    tc_tensor_nd_free(bad_stride);
    tc_tensor_nd_free(big_out2);
    tc_tensor_nd_free(big_reference);
    tc_tensor_nd_free(big_padded);
    tc_tensor_nd_free(big_out);
    tc_tensor_nd_free(big_k);
    tc_tensor_nd_free(big_in);
    tc_tensor_nd_free(reference);
    tc_tensor_nd_free(padded_in);
    tc_tensor_nd_free(multi_out);
    tc_tensor_nd_free(multi_k);
    tc_tensor_nd_free(multi_in);
    tc_tensor_nd_free(strided);
    tc_tensor_nd_free(invalid_output);
    tc_tensor_nd_free(incompatible);
    tc_tensor_nd_free(loss);
    tc_tensor_nd_free(output);
    tc_tensor_nd_free(kernel);
    tc_tensor_nd_free(input);
    return ok ? 0 : 1;
}
