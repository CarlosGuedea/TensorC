#ifndef TENSORC_NDTENSOR_H
#define TENSORC_NDTENSOR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * N-dimensional tensors used by TensorC's graph/autograd API.
 *
 * Tensors are row-major and own their data. Operation results retain their
 * parents, so a parent may safely be released before its derived tensor.
 */
typedef struct tc_tensor_nd tc_tensor_nd;

tc_tensor_nd *tc_tensor_nd_create(const size_t *shape, size_t ndim,
                                  int requires_grad);
tc_tensor_nd *tc_tensor_nd_from_data(const size_t *shape, size_t ndim,
                                     const float *data, int requires_grad);
void tc_tensor_nd_retain(tc_tensor_nd *tensor);
void tc_tensor_nd_free(tc_tensor_nd *tensor);

size_t tc_tensor_nd_numel(const tc_tensor_nd *tensor);
size_t tc_tensor_nd_ndim(const tc_tensor_nd *tensor);
const size_t *tc_tensor_nd_shape(const tc_tensor_nd *tensor);
float *tc_tensor_nd_data(tc_tensor_nd *tensor);
const float *tc_tensor_nd_const_data(const tc_tensor_nd *tensor);
const float *tc_tensor_nd_grad(const tc_tensor_nd *tensor);
int tc_tensor_nd_requires_grad(const tc_tensor_nd *tensor);
void tc_tensor_nd_zero_grad(tc_tensor_nd *tensor);

/* Reshape is metadata-only and preserves the number of elements. */
int tc_tensor_nd_reshape(tc_tensor_nd *tensor, const size_t *shape,
                         size_t ndim);

/* Elementwise operations support trailing-dimension broadcasting. */
tc_tensor_nd *tc_tensor_nd_add(const tc_tensor_nd *a, const tc_tensor_nd *b);
tc_tensor_nd *tc_tensor_nd_mul(const tc_tensor_nd *a, const tc_tensor_nd *b);
/* Matrix multiplication is currently defined for two rank-2 tensors. */
tc_tensor_nd *tc_tensor_nd_matmul(const tc_tensor_nd *a,
                                  const tc_tensor_nd *b);
/*
 * Conv2D options: strides must be >= 1 and padding is symmetric zero
 * padding applied to the input's height and width.
 */
typedef struct {
    size_t stride_h;
    size_t stride_w;
    size_t pad_h;
    size_t pad_w;
} tc_conv2d_options;

/* Default options: stride 1 in both axes and no padding. */
#define TC_CONV2D_DEFAULT ((tc_conv2d_options){1, 1, 0, 0})

/* Valid 2D cross-correlation: input [N,C,H,W], kernel [K,C,KH,KW]. */
tc_tensor_nd *tc_tensor_nd_conv2d(const tc_tensor_nd *input,
                                   const tc_tensor_nd *kernel);
/*
 * 2D cross-correlation with configurable stride and zero padding.
 * Output shape is
 *   [N, K, (H + 2*pad_h - KH)/stride_h + 1, (W + 2*pad_w - KW)/stride_w + 1]
 * and must be strictly positive in every dimension.
 */
tc_tensor_nd *tc_tensor_nd_conv2d_ex(const tc_tensor_nd *input,
                                     const tc_tensor_nd *kernel,
                                     tc_conv2d_options options);
tc_tensor_nd *tc_tensor_nd_relu(const tc_tensor_nd *input);
tc_tensor_nd *tc_tensor_nd_sum(const tc_tensor_nd *input);
tc_tensor_nd *tc_tensor_nd_mse_loss(const tc_tensor_nd *prediction,
                                    const tc_tensor_nd *target);

/* Runs reverse-mode autodiff from a scalar loss. Returns 1 on success. */
int tc_tensor_nd_backward(tc_tensor_nd *loss);
void tc_tensor_nd_print(const tc_tensor_nd *tensor, const char *name);

#ifdef __cplusplus
}
#endif

#endif
