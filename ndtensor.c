#include <tensorc/ndtensor.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef TENSORC_HAS_ACCELERATE
#include <limits.h>
#include <Accelerate/Accelerate.h>
#endif

typedef enum {
    TC_ND_OP_LEAF = 0,
    TC_ND_OP_ADD,
    TC_ND_OP_MUL,
    TC_ND_OP_MATMUL,
    TC_ND_OP_CONV2D,
    TC_ND_OP_RELU,
    TC_ND_OP_SUM,
    TC_ND_OP_MSE,
    TC_ND_OP_RESHAPE,
    TC_ND_OP_MAXPOOL2D,
    TC_ND_OP_SOFTMAX,
    TC_ND_OP_CROSS_ENTROPY,
    TC_ND_OP_SIGMOID,
    TC_ND_OP_TANH,
    TC_ND_OP_SLICE,
    TC_ND_OP_CONCAT,
    TC_ND_OP_LAYERNORM,
    TC_ND_OP_TRANSPOSE,
    TC_ND_OP_ATTENTION
} tc_nd_op;

struct tc_tensor_nd {
    size_t ndim;
    size_t *shape;
    size_t *strides;
    size_t size;
    float *data;
    float *grad;
    unsigned int refcount;
    int requires_grad;
    tc_nd_op op;
    tc_conv2d_options conv_options;
    size_t parent_count;
    tc_tensor_nd **parents;
    /* Parámetros de slice_axis: eje y desplazamiento de inicio. */
    size_t slice_axis;
    size_t slice_start;
    /* Máscara causal de la atención (1 = no atender al futuro). */
    int attn_causal;
};

static int checked_product(const size_t *shape, size_t ndim, size_t *result) {
    size_t product = 1;
    if (!shape || ndim == 0) return 0;
    for (size_t i = 0; i < ndim; ++i) {
        if (shape[i] == 0 || product > (size_t)-1 / shape[i]) return 0;
        product *= shape[i];
    }
    if (product > (size_t)-1 / sizeof(float)) return 0;
    *result = product;
    return 1;
}

static void build_strides(tc_tensor_nd *tensor) {
    tensor->strides[tensor->ndim - 1] = 1;
    for (size_t i = tensor->ndim - 1; i > 0; --i)
        tensor->strides[i - 1] = tensor->strides[i] * tensor->shape[i];
}

static tc_tensor_nd *tensor_alloc(const size_t *shape, size_t ndim,
                                  int requires_grad, int zero_data) {
    size_t size = 0;
    if (!checked_product(shape, ndim, &size)) return NULL;

    tc_tensor_nd *tensor = (tc_tensor_nd *)calloc(1, sizeof(*tensor));
    if (!tensor) return NULL;
    tensor->shape = (size_t *)malloc(ndim * sizeof(size_t));
    tensor->strides = (size_t *)malloc(ndim * sizeof(size_t));
    tensor->data = zero_data ? (float *)calloc(size, sizeof(float))
                             : (float *)malloc(size * sizeof(float));
    tensor->grad = requires_grad ? (float *)calloc(size, sizeof(float)) : NULL;
    tensor->ndim = ndim;
    tensor->size = size;
    tensor->refcount = 1;
    tensor->requires_grad = requires_grad != 0;
    tensor->op = TC_ND_OP_LEAF;
    if (!tensor->shape || !tensor->strides || !tensor->data ||
        (requires_grad && !tensor->grad)) {
        free(tensor->shape);
        free(tensor->strides);
        free(tensor->data);
        free(tensor->grad);
        free(tensor);
        return NULL;
    }
    memcpy(tensor->shape, shape, ndim * sizeof(size_t));
    build_strides(tensor);
    return tensor;
}

tc_tensor_nd *tc_tensor_nd_create(const size_t *shape, size_t ndim,
                                  int requires_grad) {
    return tensor_alloc(shape, ndim, requires_grad, 1);
}

tc_tensor_nd *tc_tensor_nd_from_data(const size_t *shape, size_t ndim,
                                     const float *data, int requires_grad) {
    if (!data) return NULL;
    tc_tensor_nd *tensor = tensor_alloc(shape, ndim, requires_grad, 0);
    if (!tensor) return NULL;
    memcpy(tensor->data, data, tensor->size * sizeof(float));
    return tensor;
}

void tc_tensor_nd_retain(tc_tensor_nd *tensor) {
    if (tensor && tensor->refcount < (unsigned int)-1) ++tensor->refcount;
}

void tc_tensor_nd_free(tc_tensor_nd *tensor) {
    if (!tensor || tensor->refcount == 0) return;
    if (--tensor->refcount != 0) return;
    for (size_t i = 0; i < tensor->parent_count; ++i)
        tc_tensor_nd_free(tensor->parents[i]);
    free(tensor->parents);
    free(tensor->shape);
    free(tensor->strides);
    free(tensor->data);
    free(tensor->grad);
    free(tensor);
}

size_t tc_tensor_nd_numel(const tc_tensor_nd *tensor) {
    return tensor ? tensor->size : 0;
}

size_t tc_tensor_nd_ndim(const tc_tensor_nd *tensor) {
    return tensor ? tensor->ndim : 0;
}

const size_t *tc_tensor_nd_shape(const tc_tensor_nd *tensor) {
    return tensor ? tensor->shape : NULL;
}

float *tc_tensor_nd_data(tc_tensor_nd *tensor) {
    return tensor ? tensor->data : NULL;
}

const float *tc_tensor_nd_const_data(const tc_tensor_nd *tensor) {
    return tensor ? tensor->data : NULL;
}

const float *tc_tensor_nd_grad(const tc_tensor_nd *tensor) {
    return tensor ? tensor->grad : NULL;
}

int tc_tensor_nd_requires_grad(const tc_tensor_nd *tensor) {
    return tensor ? tensor->requires_grad : 0;
}

void tc_tensor_nd_zero_grad(tc_tensor_nd *tensor) {
    if (tensor && tensor->grad) memset(tensor->grad, 0, tensor->size * sizeof(float));
}

int tc_tensor_nd_reshape(tc_tensor_nd *tensor, const size_t *shape,
                         size_t ndim) {
    size_t size = 0;
    if (!tensor || tensor->parent_count != 0 ||
        !checked_product(shape, ndim, &size) || size != tensor->size) return 0;
    size_t *new_shape = (size_t *)malloc(ndim * sizeof(size_t));
    size_t *new_strides = (size_t *)malloc(ndim * sizeof(size_t));
    if (!new_shape || !new_strides) {
        free(new_shape);
        free(new_strides);
        return 0;
    }
    memcpy(new_shape, shape, ndim * sizeof(size_t));
    free(tensor->shape);
    free(tensor->strides);
    tensor->shape = new_shape;
    tensor->strides = new_strides;
    tensor->ndim = ndim;
    build_strides(tensor);
    return 1;
}

static tc_tensor_nd *operation_result(const size_t *shape, size_t ndim,
                                      tc_nd_op op, tc_tensor_nd **parents,
                                      size_t parent_count) {
    int requires_grad = 0;
    for (size_t i = 0; i < parent_count; ++i)
        requires_grad = requires_grad || parents[i]->requires_grad;
    tc_tensor_nd *result = tensor_alloc(shape, ndim, requires_grad, 1);
    if (!result) return NULL;
    result->op = op;
    result->parent_count = parent_count;
    if (parent_count > 0) {
        result->parents = (tc_tensor_nd **)malloc(parent_count *
                                                  sizeof(*parents));
        if (!result->parents) {
            tc_tensor_nd_free(result);
            return NULL;
        }
        for (size_t i = 0; i < parent_count; ++i) {
            result->parents[i] = parents[i];
            tc_tensor_nd_retain(parents[i]);
        }
    }
    return result;
}

/* Versión de dos padres para las operaciones binarias existentes.
 * Acepta punteros const porque solo necesitan retenerse (refcount). */
static tc_tensor_nd *binary_result(const size_t *shape, size_t ndim,
                                   tc_nd_op op, const tc_tensor_nd *a,
                                   const tc_tensor_nd *b) {
    tc_tensor_nd *parents[] = {(tc_tensor_nd *)a, (tc_tensor_nd *)b};
    return operation_result(shape, ndim, op, parents, 2);
}

static tc_tensor_nd *unary_result(const size_t *shape, size_t ndim,
                                  tc_nd_op op, const tc_tensor_nd *a) {
    tc_tensor_nd *parent = (tc_tensor_nd *)a;
    return operation_result(shape, ndim, op, &parent, 1);
}

static int same_shape(const tc_tensor_nd *a, const tc_tensor_nd *b) {
    if (!a || !b || a->ndim != b->ndim || a->size != b->size) return 0;
    return memcmp(a->shape, b->shape, a->ndim * sizeof(size_t)) == 0;
}

/* NumPy-style trailing-dimension broadcasting for elementwise operations. */
static int broadcast_shape(const tc_tensor_nd *a, const tc_tensor_nd *b,
                           size_t **shape_out, size_t *ndim_out) {
    if (!a || !b || !shape_out || !ndim_out) return 0;
    const size_t ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
    if (ndim > (size_t)-1 / sizeof(size_t)) return 0;
    size_t *shape = (size_t *)malloc(ndim * sizeof(*shape));
    if (!shape) return 0;

    const size_t a_offset = ndim - a->ndim;
    const size_t b_offset = ndim - b->ndim;
    for (size_t axis = 0; axis < ndim; ++axis) {
        const size_t a_dim = axis < a_offset ? 1 : a->shape[axis - a_offset];
        const size_t b_dim = axis < b_offset ? 1 : b->shape[axis - b_offset];
        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
            free(shape);
            return 0;
        }
        shape[axis] = a_dim > b_dim ? a_dim : b_dim;
    }
    *shape_out = shape;
    *ndim_out = ndim;
    return 1;
}

static size_t broadcast_index(const tc_tensor_nd *input,
                              const tc_tensor_nd *output,
                              size_t output_index) {
    const size_t offset = output->ndim - input->ndim;
    size_t input_index = 0;
    for (size_t axis = 0; axis < input->ndim; ++axis) {
        const size_t output_axis = axis + offset;
        const size_t coordinate =
            (output_index / output->strides[output_axis]) % output->shape[output_axis];
        if (input->shape[axis] != 1)
            input_index += coordinate * input->strides[axis];
    }
    return input_index;
}

tc_tensor_nd *tc_tensor_nd_add(const tc_tensor_nd *a, const tc_tensor_nd *b) {
    size_t *shape = NULL;
    size_t ndim = 0;
    if (!broadcast_shape(a, b, &shape, &ndim)) return NULL;
    tc_tensor_nd *result = binary_result(shape, ndim, TC_ND_OP_ADD,
                                            a, b);
    free(shape);
    if (!result) return NULL;
    for (size_t i = 0; i < result->size; ++i) {
        const size_t a_index = broadcast_index(a, result, i);
        const size_t b_index = broadcast_index(b, result, i);
        result->data[i] = a->data[a_index] + b->data[b_index];
    }
    return result;
}

tc_tensor_nd *tc_tensor_nd_mul(const tc_tensor_nd *a, const tc_tensor_nd *b) {
    size_t *shape = NULL;
    size_t ndim = 0;
    if (!broadcast_shape(a, b, &shape, &ndim)) return NULL;
    tc_tensor_nd *result = binary_result(shape, ndim, TC_ND_OP_MUL,
                                            a, b);
    free(shape);
    if (!result) return NULL;
    for (size_t i = 0; i < result->size; ++i) {
        const size_t a_index = broadcast_index(a, result, i);
        const size_t b_index = broadcast_index(b, result, i);
        result->data[i] = a->data[a_index] * b->data[b_index];
    }
    return result;
}

tc_tensor_nd *tc_tensor_nd_matmul(const tc_tensor_nd *a, const tc_tensor_nd *b) {
    if (!a || !b || a->ndim != 2 || b->ndim != 2 || a->shape[1] != b->shape[0]) return NULL;
    const size_t shape[] = {a->shape[0], b->shape[1]};
    tc_tensor_nd *result = binary_result(shape, 2, TC_ND_OP_MATMUL,
                                            a, b);
    if (!result) return NULL;
    for (size_t i = 0; i < a->shape[0]; ++i)
        for (size_t k = 0; k < a->shape[1]; ++k)
            for (size_t j = 0; j < b->shape[1]; ++j)
                result->data[i * b->shape[1] + j] +=
                    a->data[i * a->shape[1] + k] * b->data[k * b->shape[1] + j];
    return result;
}

#ifdef TENSORC_HAS_ACCELERATE
/*
 * Builds the im2col matrix for one image: row (oh,ow) holds the flattened
 * window [C*KH*KW] with zero padding. Layout: col[(oh*ow + ow) * cols + k].
 */
static void conv2d_build_col_image(const tc_tensor_nd *input,
                                   tc_conv2d_options options, size_t n,
                                   size_t output_height, size_t output_width,
                                   size_t kernel_height, size_t kernel_width,
                                   float *col) {
    const size_t input_channels = input->shape[1];
    const size_t input_height = input->shape[2];
    const size_t input_width = input->shape[3];
    const size_t cols = input_channels * kernel_height * kernel_width;
    for (size_t oh = 0; oh < output_height; ++oh)
        for (size_t ow = 0; ow < output_width; ++ow) {
            float *row = col + (oh * output_width + ow) * cols;
            for (size_t ic = 0; ic < input_channels; ++ic)
                for (size_t kh = 0; kh < kernel_height; ++kh) {
                    const size_t input_row = oh * options.stride_h + kh;
                    const int row_valid = input_row >= options.pad_h &&
                                          input_row < input_height + options.pad_h;
                    for (size_t kw = 0; kw < kernel_width; ++kw) {
                        const size_t input_col = ow * options.stride_w + kw;
                        row[(ic * kernel_height + kh) * kernel_width + kw] =
                            (row_valid && input_col >= options.pad_w &&
                             input_col < input_width + options.pad_w)
                            ? input->data[((n * input_channels + ic) * input_height +
                                           input_row - options.pad_h) * input_width +
                                          input_col - options.pad_w]
                            : 0.0f;
                    }
                }
        }
}
#endif

static void conv2d_forward_naive(const tc_tensor_nd *input,
                                 const tc_tensor_nd *kernel,
                                 tc_conv2d_options options,
                                 tc_tensor_nd *result) {
    const size_t batch = input->shape[0];
    const size_t input_channels = input->shape[1];
    const size_t input_height = input->shape[2];
    const size_t input_width = input->shape[3];
    const size_t output_channels = kernel->shape[0];
    const size_t kernel_height = kernel->shape[2];
    const size_t kernel_width = kernel->shape[3];
    const size_t output_height = result->shape[2];
    const size_t output_width = result->shape[3];

    for (size_t n = 0; n < batch; ++n)
        for (size_t oc = 0; oc < output_channels; ++oc)
            for (size_t oh = 0; oh < output_height; ++oh) {
                const size_t row_offset = oh * options.stride_h;
                for (size_t ow = 0; ow < output_width; ++ow) {
                    const size_t col_offset = ow * options.stride_w;
                    float value = 0.0f;
                    for (size_t ic = 0; ic < input_channels; ++ic)
                        for (size_t kh = 0; kh < kernel_height; ++kh) {
                            const size_t input_row = row_offset + kh;
                            if (input_row < options.pad_h ||
                                input_row >= input_height + options.pad_h)
                                continue;
                            for (size_t kw = 0; kw < kernel_width; ++kw) {
                                const size_t input_col = col_offset + kw;
                                if (input_col < options.pad_w ||
                                    input_col >= input_width + options.pad_w)
                                    continue;
                                const size_t input_index =
                                    (((n * input_channels + ic) * input_height +
                                      input_row - options.pad_h) *
                                     input_width + input_col - options.pad_w);
                                const size_t kernel_index =
                                    (((oc * input_channels + ic) * kernel_height + kh) *
                                     kernel_width + kw);
                                value += input->data[input_index] *
                                         kernel->data[kernel_index];
                            }
                        }
                    const size_t output_index =
                        (((n * output_channels + oc) * output_height + oh) * output_width + ow);
                    result->data[output_index] = value;
                }
            }
}

#ifdef TENSORC_HAS_ACCELERATE
/*
 * Ruta rápida: im2col + GEMM de Accelerate, una imagen a la vez. La salida
 * del GEMM cae directamente en el layout [N, K, OH, OW] de cada imagen.
 * Devuelve 1 si procesó la convolución; 0 pide el camino genérico.
 */
static int conv2d_accelerate_forward(const tc_tensor_nd *input,
                                     const tc_tensor_nd *kernel,
                                     tc_conv2d_options options,
                                     tc_tensor_nd *result) {
    const size_t batch = input->shape[0];
    const size_t input_channels = input->shape[1];
    const size_t kernel_height = kernel->shape[2];
    const size_t kernel_width = kernel->shape[3];
    const size_t output_channels = kernel->shape[0];
    const size_t output_height = result->shape[2];
    const size_t output_width = result->shape[3];
    const size_t rows = output_height * output_width;
    const size_t depth = input_channels * kernel_height * kernel_width;

    if (rows == 0 || depth == 0 || rows > (size_t)INT_MAX ||
        depth > (size_t)INT_MAX || output_channels > (size_t)INT_MAX)
        return 0;
    const size_t output_size = batch * output_channels * rows;
    if (output_size > ((size_t)-1) / depth ||
        output_size * depth < ((size_t)1 << 16))
        return 0;

    float *col = (float *)malloc(rows * depth * sizeof(float));
    if (!col) return 0;
    for (size_t n = 0; n < batch; ++n) {
        conv2d_build_col_image(input, options, n, output_height, output_width,
                               kernel_height, kernel_width, col);
        float *out_n = result->data + n * output_channels * rows;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (int)output_channels, (int)rows, (int)depth,
                    1.0f, kernel->data, (int)depth,
                    col, (int)depth,
                    0.0f, out_n, (int)rows);
    }
    free(col);
    return 1;
}
#endif

tc_tensor_nd *tc_tensor_nd_conv2d(const tc_tensor_nd *input,
                                   const tc_tensor_nd *kernel) {
    return tc_tensor_nd_conv2d_ex(input, kernel, TC_CONV2D_DEFAULT);
}

tc_tensor_nd *tc_tensor_nd_conv2d_ex(const tc_tensor_nd *input,
                                     const tc_tensor_nd *kernel,
                                     tc_conv2d_options options) {
    if (!input || !kernel || input->ndim != 4 || kernel->ndim != 4 ||
        input->shape[1] != kernel->shape[1] || options.stride_h == 0 ||
        options.stride_w == 0 ||
        options.pad_h > ((size_t)-1 - input->shape[2]) / 2 ||
        options.pad_w > ((size_t)-1 - input->shape[3]) / 2)
        return NULL;

    const size_t padded_height = input->shape[2] + 2 * options.pad_h;
    const size_t padded_width = input->shape[3] + 2 * options.pad_w;
    if (kernel->shape[2] > padded_height || kernel->shape[3] > padded_width)
        return NULL;

    const size_t output_height = (padded_height - kernel->shape[2]) / options.stride_h + 1;
    const size_t output_width = (padded_width - kernel->shape[3]) / options.stride_w + 1;
    const size_t output_shape[] = {
        input->shape[0], kernel->shape[0], output_height, output_width
    };
    tc_tensor_nd *result = binary_result(output_shape, 4, TC_ND_OP_CONV2D,
                                            input,
                                            kernel);
    if (!result) return NULL;
    result->conv_options = options;

#ifdef TENSORC_HAS_ACCELERATE
    if (conv2d_accelerate_forward(input, kernel, options, result))
        return result;
#endif
    conv2d_forward_naive(input, kernel, options, result);
    return result;
}

tc_tensor_nd *tc_tensor_nd_relu(const tc_tensor_nd *input) {
    if (!input) return NULL;
    tc_tensor_nd *result = unary_result(input->shape, input->ndim, TC_ND_OP_RELU,
                                            input);
    if (!result) return NULL;
    for (size_t i = 0; i < input->size; ++i)
        result->data[i] = input->data[i] > 0.0f ? input->data[i] : 0.0f;
    return result;
}

tc_tensor_nd *tc_tensor_nd_sigmoid(const tc_tensor_nd *input) {
    if (!input) return NULL;
    tc_tensor_nd *result = unary_result(input->shape, input->ndim,
                                            TC_ND_OP_SIGMOID,
                                            input);
    if (!result) return NULL;
    for (size_t i = 0; i < input->size; ++i) {
        /* Rama estable según el signo: evita exp() de argumento positivo. */
        const float x = input->data[i];
        result->data[i] = x >= 0.0f
            ? 1.0f / (1.0f + expf(-x))
            : expf(x) / (1.0f + expf(x));
    }
    return result;
}

tc_tensor_nd *tc_tensor_nd_tanh(const tc_tensor_nd *input) {
    if (!input) return NULL;
    tc_tensor_nd *result = unary_result(input->shape, input->ndim,
                                            TC_ND_OP_TANH,
                                            input);
    if (!result) return NULL;
    for (size_t i = 0; i < input->size; ++i)
        result->data[i] = tanhf(input->data[i]);
    return result;
}

tc_tensor_nd *tc_tensor_nd_softmax(const tc_tensor_nd *input) {
    if (!input || input->ndim == 0 || input->shape[input->ndim - 1] == 0)
        return NULL;
    tc_tensor_nd *result = unary_result(input->shape, input->ndim,
                                            TC_ND_OP_SOFTMAX,
                                            input);
    if (!result) return NULL;
    const size_t classes = input->shape[input->ndim - 1];
    const size_t rows = input->size / classes;
    for (size_t r = 0; r < rows; ++r) {
        const float *src = input->data + r * classes;
        float *dst = result->data + r * classes;
        float max_value = src[0];
        for (size_t j = 1; j < classes; ++j)
            if (src[j] > max_value) max_value = src[j];
        float sum = 0.0f;
        for (size_t j = 0; j < classes; ++j) {
            dst[j] = expf(src[j] - max_value);
            sum += dst[j];
        }
        for (size_t j = 0; j < classes; ++j)
            dst[j] /= sum;
    }
    return result;
}

tc_tensor_nd *tc_tensor_nd_cross_entropy(const tc_tensor_nd *prediction,
                                         const tc_tensor_nd *target) {
    if (!same_shape(prediction, target) || prediction->ndim == 0 ||
        prediction->shape[prediction->ndim - 1] == 0)
        return NULL;
    const size_t classes = prediction->shape[prediction->ndim - 1];
    const size_t rows = prediction->size / classes;
    const size_t shape[] = {1};
    tc_tensor_nd *result = binary_result(shape, 1, TC_ND_OP_CROSS_ENTROPY,
                                            prediction,
                                            target);
    if (!result) return NULL;
    for (size_t r = 0; r < rows; ++r)
        for (size_t j = 0; j < classes; ++j) {
            const float p = prediction->data[r * classes + j];
            result->data[0] -= target->data[r * classes + j] *
                               logf(p > 1e-7f ? p : 1e-7f);
        }
    result->data[0] /= (float)rows;
    return result;
}

tc_tensor_nd *tc_tensor_nd_reshape_to(const tc_tensor_nd *input,
                                      const size_t *shape, size_t ndim) {
    size_t size = 0;
    if (!input || !checked_product(shape, ndim, &size) || size != input->size)
        return NULL;
    tc_tensor_nd *result = unary_result(shape, ndim, TC_ND_OP_RESHAPE,
                                            input);
    if (!result) return NULL;
    memcpy(result->data, input->data, input->size * sizeof(float));
    return result;
}

tc_tensor_nd *tc_tensor_nd_maxpool2d(const tc_tensor_nd *input,
                                     size_t pool_h, size_t pool_w) {
    if (!input || input->ndim != 4 || pool_h == 0 || pool_w == 0 ||
        input->shape[2] % pool_h != 0 || input->shape[3] % pool_w != 0)
        return NULL;
    const size_t output_shape[] = {
        input->shape[0], input->shape[1],
        input->shape[2] / pool_h, input->shape[3] / pool_w
    };
    tc_tensor_nd *result = unary_result(output_shape, 4, TC_ND_OP_MAXPOOL2D,
                                            input);
    if (!result) return NULL;

    const size_t batch = input->shape[0];
    const size_t channels = input->shape[1];
    const size_t input_height = input->shape[2];
    const size_t input_width = input->shape[3];
    const size_t output_height = result->shape[2];
    const size_t output_width = result->shape[3];
    for (size_t n = 0; n < batch; ++n)
        for (size_t c = 0; c < channels; ++c)
            for (size_t oh = 0; oh < output_height; ++oh)
                for (size_t ow = 0; ow < output_width; ++ow) {
                    float best = -INFINITY;
                    const size_t row0 = oh * pool_h;
                    const size_t col0 = ow * pool_w;
                    for (size_t ph = 0; ph < pool_h; ++ph)
                        for (size_t pw = 0; pw < pool_w; ++pw) {
                            const float value = input->data[
                                (((n * channels + c) * input_height + row0 + ph) *
                                 input_width + col0 + pw)];
                            if (value > best) best = value;
                        }
                    result->data[(((n * channels + c) * output_height + oh) *
                                  output_width + ow)] = best;
                }
    return result;
}

tc_tensor_nd *tc_tensor_nd_sum(const tc_tensor_nd *input) {
    if (!input) return NULL;
    const size_t shape[] = {1};
    tc_tensor_nd *result = unary_result(shape, 1, TC_ND_OP_SUM,
                                            input);
    if (!result) return NULL;
    for (size_t i = 0; i < input->size; ++i) result->data[0] += input->data[i];
    return result;
}

tc_tensor_nd *tc_tensor_nd_mse_loss(const tc_tensor_nd *prediction,
                                    const tc_tensor_nd *target) {
    if (!same_shape(prediction, target)) return NULL;
    const size_t shape[] = {1};
    tc_tensor_nd *result = binary_result(shape, 1, TC_ND_OP_MSE,
                                            prediction,
                                            target);
    if (!result) return NULL;
    for (size_t i = 0; i < prediction->size; ++i) {
        const float error = prediction->data[i] - target->data[i];
        result->data[0] += error * error;
    }
    result->data[0] /= (float)prediction->size;
    return result;
}

/*
 * Slice contiguo a lo largo de un eje: la salida conserva el rango y copia
 * shape[axis] = length. Requiere start + length <= input->shape[axis].
 */
tc_tensor_nd *tc_tensor_nd_slice_axis(const tc_tensor_nd *input, size_t axis,
                                      size_t start, size_t length) {
    if (!input || axis >= input->ndim || length == 0 ||
        start > input->shape[axis] || length > input->shape[axis] - start)
        return NULL;
    size_t shape[8];
    for (size_t i = 0; i < input->ndim; ++i)
        shape[i] = i == axis ? length : input->shape[i];
    tc_tensor_nd *result = unary_result(shape, input->ndim, TC_ND_OP_SLICE,
                                        input);
    if (!result) return NULL;
    result->slice_axis = axis;
    result->slice_start = start;
    const size_t inner = input->strides[axis];
    const size_t outer = input->size / (input->shape[axis] * inner);
    const size_t block = length * inner;
    for (size_t o = 0; o < outer; ++o) {
        const size_t src_base = o * input->shape[axis] * inner + start * inner;
        memcpy(result->data + o * block, input->data + src_base,
               block * sizeof(float));
    }
    return result;
}

/*
 * Concatena N tensores a lo largo de un eje. Todos deben tener el mismo
 * rango y coincidir en todas las dimensiones excepto en `axis`.
 */
tc_tensor_nd *tc_tensor_nd_concat(const tc_tensor_nd *const *inputs,
                                  size_t count, size_t axis) {
    if (!inputs || count == 0 || !inputs[0] || axis >= inputs[0]->ndim)
        return NULL;
    /* Caso trivial: un solo tensor pasa derecho (reshape metadata). */
    if (count == 1) {
        tc_tensor_nd *result =
            unary_result(inputs[0]->shape, inputs[0]->ndim, TC_ND_OP_RESHAPE,
                         inputs[0]);
        if (result) memcpy(result->data, inputs[0]->data,
                           inputs[0]->size * sizeof(float));
        return result;
    }
    const size_t ndim = inputs[0]->ndim;
    for (size_t i = 1; i < count; ++i) {
        if (!inputs[i] || inputs[i]->ndim != ndim) return NULL;
        for (size_t d = 0; d < ndim; ++d)
            if (d != axis && inputs[i]->shape[d] != inputs[0]->shape[d])
                return NULL;
    }
    size_t shape[8], axis_total = 0;
    for (size_t d = 0; d < ndim; ++d) shape[d] = inputs[0]->shape[d];
    for (size_t i = 0; i < count; ++i) axis_total += inputs[i]->shape[axis];
    shape[axis] = axis_total;

    tc_tensor_nd **parents = (tc_tensor_nd **)malloc(count * sizeof(*parents));
    if (!parents) return NULL;
    for (size_t i = 0; i < count; ++i)
        parents[i] = (tc_tensor_nd *)inputs[i];
    tc_tensor_nd *result = operation_result(shape, ndim, TC_ND_OP_CONCAT,
                                            parents, count);
    free(parents);
    if (!result) return NULL;

    /* Copia bloque a bloque: cada tensor aporta su tramo del eje. */
    const size_t inner = inputs[0]->strides[axis];
    const size_t outer = inputs[0]->size / (inputs[0]->shape[axis] * inner);
    size_t axis_offset = 0;
    for (size_t i = 0; i < count; ++i) {
        const size_t block = inputs[i]->shape[axis] * inner;
        for (size_t o = 0; o < outer; ++o)
            memcpy(result->data + o * axis_total * inner + axis_offset * inner,
                   inputs[i]->data + o * block, block * sizeof(float));
        axis_offset += inputs[i]->shape[axis];
    }
    return result;
}

/*
 * Layer normalization sobre la última dimensión. La salida tiene la forma
 * de la entrada; cada fila (tramo de la última dimensión) queda con media
 * 0 y varianza 1 (sesgada). La transformación afín (gamma/beta) se compone
 * con mul/add estándar.
 */
tc_tensor_nd *tc_tensor_nd_layer_norm(const tc_tensor_nd *input) {
    if (!input || input->ndim == 0 || input->shape[input->ndim - 1] == 0)
        return NULL;
    const size_t width = input->shape[input->ndim - 1];
    const size_t rows = input->size / width;
    tc_tensor_nd *result = unary_result(input->shape, input->ndim,
                                        TC_ND_OP_LAYERNORM, input);
    if (!result) return NULL;
    for (size_t r = 0; r < rows; ++r) {
        const float *src = input->data + r * width;
        float *dst = result->data + r * width;
        float mean = 0.0f;
        for (size_t j = 0; j < width; ++j) mean += src[j];
        mean /= (float)width;
        float variance = 0.0f;
        for (size_t j = 0; j < width; ++j) {
            const float d = src[j] - mean;
            variance += d * d;
            dst[j] = d;
        }
        const float inv_std = 1.0f / sqrtf(variance / (float)width + 1e-5f);
        for (size_t j = 0; j < width; ++j) dst[j] *= inv_std;
    }
    return result;
}

/*
 * Transposición de una matriz (rango 2): [M, N] -> [N, M]. Es la pieza
 * que falta para proyectar en LSTM (x Wᵀ) y para Q Kᵀ en atención.
 */
tc_tensor_nd *tc_tensor_nd_transpose(const tc_tensor_nd *input) {
    if (!input || input->ndim != 2) return NULL;
    const size_t shape[] = {input->shape[1], input->shape[0]};
    tc_tensor_nd *result = unary_result(shape, 2, TC_ND_OP_TRANSPOSE, input);
    if (!result) return NULL;
    for (size_t i = 0; i < input->shape[0]; ++i)
        for (size_t j = 0; j < input->shape[1]; ++j)
            result->data[j * input->shape[0] + i] = input->data[i * input->shape[1] + j];
    return result;
}

/*
 * Atención de producto punto escalado (single-head) sobre tensores
 * [batch, seq, depth]: softmax((Q Kᵀ) / sqrt(depth) + máscara) V.
 * Con causal=1 pone -inf en la parte estrictamente superior, de modo que
 * cada posición solo atiende a sí misma y a las anteriores. Las
 * proyecciones Q/K/V son responsabilidad del caller (p. ej. capas densas).
 */
tc_tensor_nd *tc_tensor_nd_attention(const tc_tensor_nd *query,
                                     const tc_tensor_nd *key,
                                     const tc_tensor_nd *value,
                                     int causal) {
    if (!query || !key || !value || query->ndim != 3 || key->ndim != 3 ||
        value->ndim != 3 || query->shape[0] != key->shape[0] ||
        query->shape[0] != value->shape[0] ||
        query->shape[2] != key->shape[2] ||
        key->shape[1] != value->shape[1] ||
        query->shape[1] == 0 || key->shape[1] == 0)
        return NULL;
    const size_t shape[] = {query->shape[0], query->shape[1], value->shape[2]};
    tc_tensor_nd *parents[] = {(tc_tensor_nd *)query, (tc_tensor_nd *)key,
                               (tc_tensor_nd *)value};
    tc_tensor_nd *result = operation_result(shape, 3, TC_ND_OP_ATTENTION,
                                            parents, 3);
    if (!result) return NULL;
    result->attn_causal = causal != 0;

    const size_t batch = query->shape[0];
    const size_t q_seq = query->shape[1];
    const size_t k_seq = key->shape[1];
    const size_t depth = query->shape[2];
    const size_t v_depth = value->shape[2];
    const float scale = 1.0f / sqrtf((float)depth);
    float *scores = (float *)malloc(q_seq * k_seq * sizeof(float));
    if (!scores) {
        tc_tensor_nd_free(result);
        return NULL;
    }
    for (size_t n = 0; n < batch; ++n) {
        const float *q = query->data + n * q_seq * depth;
        const float *k = key->data + n * k_seq * depth;
        const float *v = value->data + n * k_seq * v_depth;
        float *out = result->data + n * q_seq * v_depth;
        for (size_t i = 0; i < q_seq; ++i) {
            for (size_t j = 0; j < k_seq; ++j) {
                float dot = 0.0f;
                for (size_t d = 0; d < depth; ++d)
                    dot += q[i * depth + d] * k[j * depth + d];
                scores[i * k_seq + j] = dot * scale;
            }
            if (causal)
                for (size_t j = i + 1; j < k_seq; ++j)
                    scores[i * k_seq + j] = -INFINITY;
            float max_score = scores[i * k_seq];
            for (size_t j = 1; j < k_seq; ++j)
                if (scores[i * k_seq + j] > max_score)
                    max_score = scores[i * k_seq + j];
            float sum = 0.0f;
            for (size_t j = 0; j < k_seq; ++j) {
                scores[i * k_seq + j] = expf(scores[i * k_seq + j] - max_score);
                sum += scores[i * k_seq + j];
            }
            for (size_t j = 0; j < k_seq; ++j)
                scores[i * k_seq + j] /= sum;
            for (size_t d = 0; d < v_depth; ++d) {
                float acc = 0.0f;
                for (size_t j = 0; j < k_seq; ++j)
                    acc += scores[i * k_seq + j] * v[j * v_depth + d];
                out[i * v_depth + d] = acc;
            }
        }
    }
    free(scores);
    return result;
}

#ifdef TENSORC_HAS_ACCELERATE
/*
 * Backward vía im2col + GEMM:
 *   dW       = sum_n col_n^T @ dgrad_n   -> scatter al gradiente del kernel
 *   dcol_n   = dgrad_n^T @ W             -> col2im al gradiente de la entrada
 * Devuelve 1 si procesó el nodo; 0 pide el camino genérico.
 */
static int conv2d_accelerate_backward(tc_tensor_nd *input, tc_tensor_nd *kernel,
                                      tc_tensor_nd *output_node) {
    const tc_conv2d_options options = output_node->conv_options;
    const size_t batch = input->shape[0];
    const size_t input_channels = input->shape[1];
    const size_t input_height = input->shape[2];
    const size_t input_width = input->shape[3];
    const size_t filters = kernel->shape[0];
    const size_t kernel_height = kernel->shape[2];
    const size_t kernel_width = kernel->shape[3];
    const size_t output_height = output_node->shape[2];
    const size_t output_width = output_node->shape[3];
    const size_t rows = output_height * output_width;
    const size_t depth = input_channels * kernel_height * kernel_width;

    if (rows == 0 || depth == 0 || rows > (size_t)INT_MAX ||
        depth > (size_t)INT_MAX || filters > (size_t)INT_MAX)
        return 0;
    const size_t output_size = batch * filters * rows;
    if (output_size > ((size_t)-1) / depth ||
        output_size * depth < ((size_t)1 << 16))
        return 0;

    float *col = (float *)malloc(rows * depth * sizeof(float));
    float *dw = (float *)calloc(depth * filters, sizeof(float));
    float *dcol = (float *)malloc(rows * depth * sizeof(float));
    if (!col || !dw || !dcol) {
        free(col);
        free(dw);
        free(dcol);
        return 0;
    }

    for (size_t n = 0; n < batch; ++n) {
        const float *dgrad_n = output_node->grad + n * filters * rows;
        conv2d_build_col_image(input, options, n, output_height, output_width,
                               kernel_height, kernel_width, col);
        /* dw += col_n^T @ dgrad_n, con dw [depth, filters]. */
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans,
                    (int)depth, (int)filters, (int)rows,
                    1.0f, col, (int)depth,
                    dgrad_n, (int)rows,
                    n == 0 ? 0.0f : 1.0f, dw, (int)filters);
        /* dcol_n [rows, depth] = dgrad_n^T @ W. */
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    (int)rows, (int)depth, (int)filters,
                    1.0f, dgrad_n, (int)rows,
                    kernel->data, (int)depth,
                    0.0f, dcol, (int)depth);
        /* col2im: dispersa dcol en el gradiente de la entrada. */
        if (input->requires_grad && input->grad) {
            for (size_t oh = 0; oh < output_height; ++oh) {
                const size_t row_offset = oh * options.stride_h;
                for (size_t ow = 0; ow < output_width; ++ow) {
                    const size_t col_offset = ow * options.stride_w;
                    const float *dcol_row = dcol + (oh * output_width + ow) * depth;
                    for (size_t ic = 0; ic < input_channels; ++ic)
                        for (size_t kh = 0; kh < kernel_height; ++kh) {
                            const size_t input_row = row_offset + kh;
                            if (input_row < options.pad_h ||
                                input_row >= input_height + options.pad_h)
                                continue;
                            for (size_t kw = 0; kw < kernel_width; ++kw) {
                                const size_t input_col = col_offset + kw;
                                if (input_col < options.pad_w ||
                                    input_col >= input_width + options.pad_w)
                                    continue;
                                input->grad[((n * input_channels + ic) * input_height +
                                             input_row - options.pad_h) * input_width +
                                            input_col - options.pad_w] +=
                                    dcol_row[(ic * kernel_height + kh) * kernel_width + kw];
                            }
                        }
                }
            }
        }
    }

    if (kernel->requires_grad && kernel->grad)
        for (size_t f = 0; f < filters; ++f)
            for (size_t k = 0; k < depth; ++k)
                kernel->grad[f * depth + k] += dw[k * filters + f];

    free(dcol);
    free(dw);
    free(col);
    return 1;
}
#endif

typedef struct {
    tc_tensor_nd **items;
    size_t count;
    size_t capacity;
} tensor_list;

static int list_contains(const tensor_list *list, const tc_tensor_nd *tensor) {
    for (size_t i = 0; i < list->count; ++i)
        if (list->items[i] == tensor) return 1;
    return 0;
}

static int list_push(tensor_list *list, tc_tensor_nd *tensor) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 16;
        tc_tensor_nd **items = (tc_tensor_nd **)realloc(
            list->items, capacity * sizeof(*items));
        if (!items) return 0;
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = tensor;
    return 1;
}

static int build_topology(tc_tensor_nd *tensor, tensor_list *list) {
    if (list_contains(list, tensor)) return 1;
    for (size_t i = 0; i < tensor->parent_count; ++i)
        if (!build_topology(tensor->parents[i], list)) return 0;
    return list_push(list, tensor);
}

static void add_grad(tc_tensor_nd *tensor, size_t index, float value) {
    if (tensor && tensor->requires_grad && tensor->grad)
        tensor->grad[index] += value;
}

static void backward_node(tc_tensor_nd *tensor) {
    if (!tensor || !tensor->requires_grad || !tensor->grad) return;
    tc_tensor_nd *a = tensor->parent_count > 0 ? tensor->parents[0] : NULL;
    tc_tensor_nd *b = tensor->parent_count > 1 ? tensor->parents[1] : NULL;
    switch (tensor->op) {
        case TC_ND_OP_ADD:
            for (size_t i = 0; i < tensor->size; ++i) {
                add_grad(a, broadcast_index(a, tensor, i), tensor->grad[i]);
                add_grad(b, broadcast_index(b, tensor, i), tensor->grad[i]);
            }
            break;
        case TC_ND_OP_MUL:
            for (size_t i = 0; i < tensor->size; ++i) {
                const size_t a_index = broadcast_index(a, tensor, i);
                const size_t b_index = broadcast_index(b, tensor, i);
                add_grad(a, a_index, tensor->grad[i] * b->data[b_index]);
                add_grad(b, b_index, tensor->grad[i] * a->data[a_index]);
            }
            break;
        case TC_ND_OP_MATMUL:
            for (size_t i = 0; i < a->shape[0]; ++i)
                for (size_t k = 0; k < a->shape[1]; ++k) {
                    float value = 0.0f;
                    for (size_t j = 0; j < b->shape[1]; ++j)
                        value += tensor->grad[i * b->shape[1] + j] *
                                 b->data[k * b->shape[1] + j];
                    add_grad(a, i * a->shape[1] + k, value);
                }
            for (size_t k = 0; k < b->shape[0]; ++k)
                for (size_t j = 0; j < b->shape[1]; ++j) {
                    float value = 0.0f;
                    for (size_t i = 0; i < a->shape[0]; ++i)
                        value += a->data[i * a->shape[1] + k] *
                                 tensor->grad[i * b->shape[1] + j];
                    add_grad(b, k * b->shape[1] + j, value);
                }
            break;
        case TC_ND_OP_CONV2D: {
#ifdef TENSORC_HAS_ACCELERATE
            if (conv2d_accelerate_backward(a, b, tensor)) break;
#endif
            const tc_conv2d_options options = tensor->conv_options;
            const size_t batch = a->shape[0];
            const size_t input_channels = a->shape[1];
            const size_t input_height = a->shape[2];
            const size_t input_width = a->shape[3];
            const size_t output_channels = b->shape[0];
            const size_t kernel_height = b->shape[2];
            const size_t kernel_width = b->shape[3];
            const size_t output_height = tensor->shape[2];
            const size_t output_width = tensor->shape[3];
            for (size_t n = 0; n < batch; ++n)
                for (size_t oc = 0; oc < output_channels; ++oc)
                    for (size_t oh = 0; oh < output_height; ++oh) {
                        const size_t row_offset = oh * options.stride_h;
                        for (size_t ow = 0; ow < output_width; ++ow) {
                            const size_t col_offset = ow * options.stride_w;
                            const size_t output_index =
                                (((n * output_channels + oc) * output_height + oh) *
                                 output_width + ow);
                            const float output_gradient = tensor->grad[output_index];
                            for (size_t ic = 0; ic < input_channels; ++ic)
                                for (size_t kh = 0; kh < kernel_height; ++kh) {
                                    const size_t input_row = row_offset + kh;
                                    if (input_row < options.pad_h ||
                                        input_row >= input_height + options.pad_h)
                                        continue;
                                    for (size_t kw = 0; kw < kernel_width; ++kw) {
                                        const size_t input_col = col_offset + kw;
                                        if (input_col < options.pad_w ||
                                            input_col >= input_width + options.pad_w)
                                            continue;
                                        const size_t input_index =
                                            (((n * input_channels + ic) * input_height +
                                              input_row - options.pad_h) *
                                             input_width + input_col - options.pad_w);
                                        const size_t kernel_index =
                                            (((oc * input_channels + ic) * kernel_height + kh) *
                                             kernel_width + kw);
                                        add_grad(a, input_index,
                                                 output_gradient * b->data[kernel_index]);
                                        add_grad(b, kernel_index,
                                                 output_gradient * a->data[input_index]);
                                    }
                                }
                        }
                    }
            break;
        }
        case TC_ND_OP_RELU:
            for (size_t i = 0; i < a->size; ++i)
                add_grad(a, i, a->data[i] > 0.0f ? tensor->grad[i] : 0.0f);
            break;
        case TC_ND_OP_SIGMOID:
            /* d sigmoid = y * (1 - y) evaluado en la salida. */
            for (size_t i = 0; i < a->size; ++i) {
                const float y = tensor->data[i];
                add_grad(a, i, tensor->grad[i] * y * (1.0f - y));
            }
            break;
        case TC_ND_OP_TANH:
            /* d tanh = 1 - y^2 evaluado en la salida. */
            for (size_t i = 0; i < a->size; ++i) {
                const float y = tensor->data[i];
                add_grad(a, i, tensor->grad[i] * (1.0f - y * y));
            }
            break;
        case TC_ND_OP_RESHAPE:
            for (size_t i = 0; i < a->size; ++i)
                add_grad(a, i, tensor->grad[i]);
            break;
        case TC_ND_OP_MAXPOOL2D: {
            const size_t pool_h = a->shape[2] / tensor->shape[2];
            const size_t pool_w = a->shape[3] / tensor->shape[3];
            const size_t batch = a->shape[0];
            const size_t channels = a->shape[1];
            const size_t input_height = a->shape[2];
            const size_t input_width = a->shape[3];
            const size_t output_height = tensor->shape[2];
            const size_t output_width = tensor->shape[3];
            for (size_t n = 0; n < batch; ++n)
                for (size_t c = 0; c < channels; ++c)
                    for (size_t oh = 0; oh < output_height; ++oh)
                        for (size_t ow = 0; ow < output_width; ++ow) {
                            const size_t row0 = oh * pool_h;
                            const size_t col0 = ow * pool_w;
                            size_t best_index = 0;
                            float best = -INFINITY;
                            for (size_t ph = 0; ph < pool_h; ++ph)
                                for (size_t pw = 0; pw < pool_w; ++pw) {
                                    const size_t index =
                                        (((n * channels + c) * input_height + row0 + ph) *
                                         input_width + col0 + pw);
                                    if (a->data[index] > best) {
                                        best = a->data[index];
                                        best_index = index;
                                    }
                                }
                            add_grad(a, best_index,
                                     tensor->grad[(((n * channels + c) * output_height + oh) *
                                                   output_width + ow)]);
                        }
            break;
        }
        case TC_ND_OP_SOFTMAX: {
            const size_t classes = a->shape[a->ndim - 1];
            const size_t rows = a->size / classes;
            for (size_t r = 0; r < rows; ++r) {
                const float *prob = tensor->data + r * classes;
                const float *grad_out = tensor->grad + r * classes;
                float dot = 0.0f;
                for (size_t j = 0; j < classes; ++j)
                    dot += grad_out[j] * prob[j];
                for (size_t j = 0; j < classes; ++j)
                    add_grad(a, r * classes + j,
                             prob[j] * (grad_out[j] - dot));
            }
            break;
        }
        case TC_ND_OP_CROSS_ENTROPY: {
            const size_t classes = a->shape[a->ndim - 1];
            const size_t rows = a->size / classes;
            const float scale = tensor->grad[0] / (float)rows;
            for (size_t i = 0; i < a->size; ++i) {
                const float p = a->data[i] > 1e-7f ? a->data[i] : 1e-7f;
                add_grad(a, i, -b->data[i] * scale / p);
                add_grad(b, i, -logf(p) * scale);
            }
            break;
        }
        case TC_ND_OP_SUM:
            for (size_t i = 0; i < a->size; ++i) add_grad(a, i, tensor->grad[0]);
            break;
        case TC_ND_OP_MSE:
            for (size_t i = 0; i < a->size; ++i) {
                const float value = 2.0f * (a->data[i] - b->data[i]) *
                                    tensor->grad[0] / (float)a->size;
                add_grad(a, i, value);
                add_grad(b, i, -value);
            }
            break;
        case TC_ND_OP_SLICE: {
            /* Gradiente inverso del slice: se suma en la región origen. */
            const size_t inner = a->strides[tensor->slice_axis];
            const size_t outer = a->size / (a->shape[tensor->slice_axis] * inner);
            const size_t block = tensor->shape[tensor->slice_axis] * inner;
            for (size_t o = 0; o < outer; ++o) {
                const size_t dst_base =
                    o * a->shape[tensor->slice_axis] * inner +
                    tensor->slice_start * inner;
                for (size_t j = 0; j < block; ++j)
                    add_grad(a, dst_base + j, tensor->grad[o * block + j]);
            }
            break;
        }
        case TC_ND_OP_CONCAT: {
            /* Cada padre recibe el tramo del eje que aportó. El eje es la
             * dimensión donde la forma del padre difiere de la salida. */
            size_t concat_axis = 0;
            for (size_t d = 0; d < tensor->ndim; ++d)
                if (tensor->shape[d] != a->shape[d]) concat_axis = d;
            const size_t inner = a->strides[concat_axis];
            const size_t outer = a->size / (a->shape[concat_axis] * inner);
            const float *grad = tensor->grad;
            size_t axis_offset = 0;
            for (size_t p = 0; p < tensor->parent_count; ++p) {
                tc_tensor_nd *parent = tensor->parents[p];
                const size_t block = parent->shape[concat_axis] * inner;
                for (size_t o = 0; o < outer; ++o)
                    for (size_t j = 0; j < block; ++j)
                        add_grad(parent, o * parent->shape[concat_axis] * inner + j,
                                 grad[o * tensor->shape[concat_axis] * inner +
                                      axis_offset * inner + j]);
                axis_offset += parent->shape[concat_axis];
            }
            break;
        }
        case TC_ND_OP_LAYERNORM: {
            /* dx = (dy - mean(dy) - xhat * mean(dy * xhat)) / sqrt(var + eps) */
            const size_t width = a->shape[a->ndim - 1];
            const size_t rows = a->size / width;
            for (size_t r = 0; r < rows; ++r) {
                const float *src = a->data + r * width;
                const float *dy = tensor->grad + r * width;
                float mean = 0.0f, variance = 0.0f;
                for (size_t j = 0; j < width; ++j) mean += src[j];
                mean /= (float)width;
                for (size_t j = 0; j < width; ++j) {
                    const float d = src[j] - mean;
                    variance += d * d;
                }
                const float inv_std =
                    1.0f / sqrtf(variance / (float)width + 1e-5f);
                float mean_dy = 0.0f, mean_dy_xhat = 0.0f;
                for (size_t j = 0; j < width; ++j) {
                    mean_dy += dy[j];
                    mean_dy_xhat += dy[j] * (src[j] - mean) * inv_std;
                }
                mean_dy /= (float)width;
                mean_dy_xhat /= (float)width;
                for (size_t j = 0; j < width; ++j) {
                    const float xhat = (src[j] - mean) * inv_std;
                    add_grad(a, r * width + j,
                             inv_std * (dy[j] - mean_dy - xhat * mean_dy_xhat));
                }
            }
            break;
        }
        case TC_ND_OP_TRANSPOSE:
            for (size_t i = 0; i < a->shape[0]; ++i)
                for (size_t j = 0; j < a->shape[1]; ++j)
                    add_grad(a, i * a->shape[1] + j,
                             tensor->grad[j * a->shape[0] + i]);
            break;
        case TC_ND_OP_ATTENTION: {
            /* Recomputa scores/P por bloque y aplica la regla
             * dV=PᵀdO, dS=P⊙(dP−rowsum(dP⊙P))·scale, dQ=dS·K, dK=dSᵀ·Q. */
            tc_tensor_nd *q = tensor->parents[0];
            tc_tensor_nd *k = tensor->parents[1];
            tc_tensor_nd *v = tensor->parents[2];
            const size_t batch = q->shape[0];
            const size_t q_seq = q->shape[1];
            const size_t k_seq = k->shape[1];
            const size_t depth = q->shape[2];
            const size_t v_depth = v->shape[2];
            const float scale = 1.0f / sqrtf((float)depth);
            const size_t score_count = q_seq * k_seq;
            float *scores = (float *)malloc(score_count * sizeof(float));
            float *dp = (float *)malloc(score_count * sizeof(float));
            if (!scores || !dp) {
                free(scores);
                free(dp);
                break;
            }
            for (size_t n = 0; n < batch; ++n) {
                const float *qd = q->data + n * q_seq * depth;
                const float *kd = k->data + n * k_seq * depth;
                const float *vd = v->data + n * k_seq * v_depth;
                const float *dout = tensor->grad + n * q_seq * v_depth;
                float *dq = q->grad ? q->grad + n * q_seq * depth : NULL;
                float *dk = k->grad ? k->grad + n * k_seq * depth : NULL;
                float *dv = v->grad ? v->grad + n * k_seq * v_depth : NULL;

                /* Rehacer softmax para recuperar P. */
                for (size_t i = 0; i < q_seq; ++i) {
                    for (size_t j = 0; j < k_seq; ++j) {
                        float dot = 0.0f;
                        for (size_t d = 0; d < depth; ++d)
                            dot += qd[i * depth + d] * kd[j * depth + d];
                        scores[i * k_seq + j] = dot * scale;
                    }
                    if (tensor->attn_causal)
                        for (size_t j = i + 1; j < k_seq; ++j)
                            scores[i * k_seq + j] = -INFINITY;
                    float max_score = scores[i * k_seq];
                    for (size_t j = 1; j < k_seq; ++j)
                        if (scores[i * k_seq + j] > max_score)
                            max_score = scores[i * k_seq + j];
                    float sum = 0.0f;
                    for (size_t j = 0; j < k_seq; ++j) {
                        scores[i * k_seq + j] = expf(scores[i * k_seq + j] - max_score);
                        sum += scores[i * k_seq + j];
                    }
                    for (size_t j = 0; j < k_seq; ++j)
                        scores[i * k_seq + j] /= sum;
                }

                /* dP = dO Vᵀ y su promedio ponderado por fila. */
                for (size_t i = 0; i < q_seq; ++i) {
                    float row_dot = 0.0f;
                    for (size_t j = 0; j < k_seq; ++j) {
                        float acc = 0.0f;
                        for (size_t d = 0; d < v_depth; ++d)
                            acc += dout[i * v_depth + d] * vd[j * v_depth + d];
                        dp[i * k_seq + j] = acc;
                        row_dot += acc * scores[i * k_seq + j];
                    }
                    for (size_t j = 0; j < k_seq; ++j) {
                        const float ds =
                            scores[i * k_seq + j] * (dp[i * k_seq + j] - row_dot) * scale;
                        dp[i * k_seq + j] = ds;
                    }
                }

                /* dV = Pᵀ dO. */
                for (size_t j = 0; j < k_seq; ++j)
                    for (size_t d = 0; d < v_depth; ++d) {
                        float acc = 0.0f;
                        for (size_t i = 0; i < q_seq; ++i)
                            acc += scores[i * k_seq + j] * dout[i * v_depth + d];
                        if (dv) dv[j * v_depth + d] += acc;
                    }
                /* dQ = dS K, dK = dSᵀ Q. */
                for (size_t i = 0; i < q_seq; ++i)
                    for (size_t d = 0; d < depth; ++d) {
                        float acc_q = 0.0f;
                        for (size_t j = 0; j < k_seq; ++j)
                            acc_q += dp[i * k_seq + j] * kd[j * depth + d];
                        if (dq) dq[i * depth + d] += acc_q;
                    }
                for (size_t j = 0; j < k_seq; ++j)
                    for (size_t d = 0; d < depth; ++d) {
                        float acc_k = 0.0f;
                        for (size_t i = 0; i < q_seq; ++i)
                            acc_k += dp[i * k_seq + j] * qd[i * depth + d];
                        if (dk) dk[j * depth + d] += acc_k;
                    }
            }
            free(scores);
            free(dp);
            break;
        }
        case TC_ND_OP_LEAF:
        default:
            break;
    }
}

int tc_tensor_nd_backward(tc_tensor_nd *loss) {
    if (!loss || !loss->requires_grad || loss->size != 1 || !loss->grad) return 0;
    tensor_list topology = {0};
    if (!build_topology(loss, &topology)) {
        free(topology.items);
        return 0;
    }
    for (size_t i = 0; i < topology.count; ++i)
        if (topology.items[i]->requires_grad) tc_tensor_nd_zero_grad(topology.items[i]);
    loss->grad[0] = 1.0f;
    for (size_t i = topology.count; i > 0; --i) backward_node(topology.items[i - 1]);
    free(topology.items);
    return 1;
}

void tc_tensor_nd_print(const tc_tensor_nd *tensor, const char *name) {
    if (!tensor || !tensor->data) {
        printf("%s: <tensor inválido>\n", name ? name : "tensor");
        return;
    }
    printf("%s shape=[", name ? name : "tensor");
    for (size_t i = 0; i < tensor->ndim; ++i)
        printf("%s%zu", i ? ", " : "", tensor->shape[i]);
    printf("] values=");
    for (size_t i = 0; i < tensor->size; ++i) printf("%s%.5f", i ? ", " : "[", tensor->data[i]);
    printf("]\n");
}

void tc_tensor_nd_sgd_step(tc_tensor_nd *param, float learning_rate) {
    if (!param || !param->grad) return;
    for (size_t i = 0; i < param->size; ++i)
        param->data[i] -= learning_rate * param->grad[i];
}

/* Generador xorshift32 determinista para la inicialización de capas. */
static unsigned int layer_rng_next(unsigned int *state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

struct tc_conv2d_layer {
    tc_tensor_nd *kernel;
    tc_tensor_nd *bias;
    tc_conv2d_options options;
};

tc_conv2d_layer *tc_conv2d_layer_create(size_t in_channels, size_t out_channels,
                                        size_t kernel_h, size_t kernel_w,
                                        tc_conv2d_options options,
                                        unsigned int seed) {
    if (in_channels == 0 || out_channels == 0 || kernel_h == 0 || kernel_w == 0)
        return NULL;
    tc_conv2d_layer *layer = (tc_conv2d_layer *)calloc(1, sizeof(*layer));
    if (!layer) return NULL;
    const size_t kernel_shape[] = {out_channels, in_channels, kernel_h, kernel_w};
    const size_t bias_shape[] = {1, out_channels, 1, 1};
    layer->kernel = tc_tensor_nd_create(kernel_shape, 4, 1);
    layer->bias = tc_tensor_nd_create(bias_shape, 4, 1);
    if (!layer->kernel || !layer->bias) {
        tc_conv2d_layer_free(layer);
        return NULL;
    }
    layer->options = options;

    /* Xavier uniforme: limit = sqrt(6 / (fan_in + fan_out)). */
    const float fan_in = (float)(in_channels * kernel_h * kernel_w);
    const float fan_out = (float)(out_channels * kernel_h * kernel_w);
    const float limit = sqrtf(6.0f / (fan_in + fan_out));
    unsigned int state = seed ? seed : 1u;
    float *kernel_data = tc_tensor_nd_data(layer->kernel);
    for (size_t i = 0; i < tc_tensor_nd_numel(layer->kernel); ++i) {
        const float unit = (float)layer_rng_next(&state) / 4294967296.0f;
        kernel_data[i] = (2.0f * unit - 1.0f) * limit;
    }
    return layer;
}

void tc_conv2d_layer_free(tc_conv2d_layer *layer) {
    if (!layer) return;
    tc_tensor_nd_free(layer->kernel);
    tc_tensor_nd_free(layer->bias);
    free(layer);
}

tc_tensor_nd *tc_conv2d_layer_forward(tc_conv2d_layer *layer,
                                      const tc_tensor_nd *input) {
    if (!layer || !input) return NULL;
    tc_tensor_nd *convolved = tc_tensor_nd_conv2d_ex(input, layer->kernel,
                                                     layer->options);
    if (!convolved) return NULL;
    tc_tensor_nd *output = tc_tensor_nd_add(convolved, layer->bias);
    tc_tensor_nd_free(convolved);
    return output;
}

tc_tensor_nd *tc_conv2d_layer_kernel(tc_conv2d_layer *layer) {
    return layer ? layer->kernel : NULL;
}

tc_tensor_nd *tc_conv2d_layer_bias(tc_conv2d_layer *layer) {
    return layer ? layer->bias : NULL;
}

struct tc_dense_layer {
    tc_tensor_nd *weight;
    tc_tensor_nd *bias;
};

tc_dense_layer *tc_dense_layer_create(size_t in_features, size_t out_features,
                                      unsigned int seed) {
    if (in_features == 0 || out_features == 0) return NULL;
    tc_dense_layer *layer = (tc_dense_layer *)calloc(1, sizeof(*layer));
    if (!layer) return NULL;
    const size_t weight_shape[] = {in_features, out_features};
    const size_t bias_shape[] = {1, out_features};
    layer->weight = tc_tensor_nd_create(weight_shape, 2, 1);
    layer->bias = tc_tensor_nd_create(bias_shape, 2, 1);
    if (!layer->weight || !layer->bias) {
        tc_dense_layer_free(layer);
        return NULL;
    }

    /* Xavier uniforme: limit = sqrt(6 / (fan_in + fan_out)). */
    const float fan_in = (float)in_features;
    const float fan_out = (float)out_features;
    const float limit = sqrtf(6.0f / (fan_in + fan_out));
    unsigned int state = seed ? seed : 1u;
    float *weight_data = tc_tensor_nd_data(layer->weight);
    for (size_t i = 0; i < tc_tensor_nd_numel(layer->weight); ++i) {
        const float unit = (float)layer_rng_next(&state) / 4294967296.0f;
        weight_data[i] = (2.0f * unit - 1.0f) * limit;
    }
    return layer;
}

void tc_dense_layer_free(tc_dense_layer *layer) {
    if (!layer) return;
    tc_tensor_nd_free(layer->weight);
    tc_tensor_nd_free(layer->bias);
    free(layer);
}

tc_tensor_nd *tc_dense_layer_forward(tc_dense_layer *layer,
                                     const tc_tensor_nd *input) {
    if (!layer || !input) return NULL;
    tc_tensor_nd *z = tc_tensor_nd_matmul(input, layer->weight);
    if (!z) return NULL;
    tc_tensor_nd *output = tc_tensor_nd_add(z, layer->bias);
    tc_tensor_nd_free(z);
    return output;
}

tc_tensor_nd *tc_dense_layer_weight(tc_dense_layer *layer) {
    return layer ? layer->weight : NULL;
}

tc_tensor_nd *tc_dense_layer_bias(tc_dense_layer *layer) {
    return layer ? layer->bias : NULL;
}

struct tc_tensor_nd_optimizer {
    tc_tensor_nd **params;
    float **m;
    float **v;
    size_t count;
    float learning_rate;
    unsigned long step_count;
};

tc_tensor_nd_optimizer *tc_tensor_nd_optimizer_adam(tc_tensor_nd **params,
                                                    size_t count,
                                                    float learning_rate) {
    if (!params && count > 0) return NULL;
    tc_tensor_nd_optimizer *optimizer =
        (tc_tensor_nd_optimizer *)calloc(1, sizeof(*optimizer));
    if (!optimizer) return NULL;
    optimizer->params = (tc_tensor_nd **)calloc(count, sizeof(*params));
    optimizer->m = (float **)calloc(count, sizeof(float *));
    optimizer->v = (float **)calloc(count, sizeof(float *));
    if ((count > 0) &&
        (!optimizer->params || !optimizer->m || !optimizer->v)) {
        tc_tensor_nd_optimizer_free(optimizer);
        return NULL;
    }
    optimizer->count = count;
    for (size_t i = 0; i < count; ++i) {
        optimizer->params[i] = params[i];
        const size_t numel = params[i] ? tc_tensor_nd_numel(params[i]) : 0;
        optimizer->m[i] = (float *)calloc(numel, sizeof(float));
        optimizer->v[i] = (float *)calloc(numel, sizeof(float));
        if (!optimizer->m[i] || !optimizer->v[i]) {
            tc_tensor_nd_optimizer_free(optimizer);
            return NULL;
        }
    }
    optimizer->learning_rate = learning_rate;
    optimizer->step_count = 0;
    return optimizer;
}

int tc_tensor_nd_optimizer_step(tc_tensor_nd_optimizer *optimizer) {
    if (!optimizer) return 0;
    const float beta1 = 0.9f, beta2 = 0.999f, epsilon = 1e-8f;
    optimizer->step_count += 1;
    const float bias1 = 1.0f - powf(beta1, (float)optimizer->step_count);
    const float bias2 = 1.0f - powf(beta2, (float)optimizer->step_count);
    int updated = 0;
    for (size_t i = 0; i < optimizer->count; ++i) {
        tc_tensor_nd *param = optimizer->params[i];
        if (!param || !param->grad) continue;
        float *m = optimizer->m[i];
        float *v = optimizer->v[i];
        for (size_t j = 0; j < param->size; ++j) {
            const float grad = param->grad[j];
            m[j] = beta1 * m[j] + (1.0f - beta1) * grad;
            v[j] = beta2 * v[j] + (1.0f - beta2) * grad * grad;
            const float m_hat = m[j] / bias1;
            const float v_hat = v[j] / bias2;
            param->data[j] -=
                optimizer->learning_rate * m_hat / (sqrtf(v_hat) + epsilon);
        }
        ++updated;
    }
    return updated;
}

void tc_tensor_nd_optimizer_free(tc_tensor_nd_optimizer *optimizer) {
    if (!optimizer) return;
    for (size_t i = 0; i < optimizer->count; ++i) {
        free(optimizer->m ? optimizer->m[i] : NULL);
        free(optimizer->v ? optimizer->v[i] : NULL);
    }
    free(optimizer->m);
    free(optimizer->v);
    free(optimizer->params);
    free(optimizer);
}

struct tc_lstm_layer {
    tc_tensor_nd *weight_ih; /* [4H, input] */
    tc_tensor_nd *weight_hh; /* [4H, hidden] */
    tc_tensor_nd *bias;      /* [1, 4H] */
    size_t hidden;
};

tc_lstm_layer *tc_lstm_layer_create(size_t input_features, size_t hidden,
                                    unsigned int seed) {
    if (input_features == 0 || hidden == 0) return NULL;
    tc_lstm_layer *layer = (tc_lstm_layer *)calloc(1, sizeof(*layer));
    if (!layer) return NULL;
    layer->hidden = hidden;
    const size_t w_ih_shape[] = {4 * hidden, input_features};
    const size_t w_hh_shape[] = {4 * hidden, hidden};
    const size_t bias_shape[] = {1, 4 * hidden};
    layer->weight_ih = tc_tensor_nd_create(w_ih_shape, 2, 1);
    layer->weight_hh = tc_tensor_nd_create(w_hh_shape, 2, 1);
    layer->bias = tc_tensor_nd_create(bias_shape, 2, 1);
    if (!layer->weight_ih || !layer->weight_hh || !layer->bias) {
        tc_lstm_layer_free(layer);
        return NULL;
    }

    /* Xavier uniforme para ambas matrices; sesgo en cero (como las demás
     * capas). Las puertas olvidadas se inicializan con sesgo positivo en
     * frameworks como PyTorch; aquí se mantiene cero por simplicidad y
     * determinismo. */
    const float fan_in = (float)(input_features + hidden);
    const float fan_out = (float)(2 * hidden);
    const float limit = sqrtf(6.0f / (fan_in + fan_out));
    unsigned int state = seed ? seed : 1u;
    float *w_ih = tc_tensor_nd_data(layer->weight_ih);
    for (size_t i = 0; i < tc_tensor_nd_numel(layer->weight_ih); ++i) {
        const float unit = (float)layer_rng_next(&state) / 4294967296.0f;
        w_ih[i] = (2.0f * unit - 1.0f) * limit;
    }
    float *w_hh = tc_tensor_nd_data(layer->weight_hh);
    for (size_t i = 0; i < tc_tensor_nd_numel(layer->weight_hh); ++i) {
        const float unit = (float)layer_rng_next(&state) / 4294967296.0f;
        w_hh[i] = (2.0f * unit - 1.0f) * limit;
    }
    return layer;
}

void tc_lstm_layer_free(tc_lstm_layer *layer) {
    if (!layer) return;
    tc_tensor_nd_free(layer->weight_ih);
    tc_tensor_nd_free(layer->weight_hh);
    tc_tensor_nd_free(layer->bias);
    free(layer);
}

/*
 * Forward sobre una secuencia completa. La entrada es [seq, batch, input]
 * (time-major, como en PyTorch sin batch_first) y la salida es
 * [seq, batch, hidden]. El estado inicial h0/c0 es cero. La salida retiene
 * todo el grafo desplegado; liberarla libera todos los pasos.
 *
 * Implementación: despliegue temporal en pasos de grafo que reutilizan los
 * mismos tensores de pesos (los gradientes se acumulan con +=). Por paso:
 *   gates = x W_ihᵀ + h W_hhᵀ + b  (una matmul por matriz)
 *   i, f, o = sigmoid(gates[:, 0:H], [H:2H], [3H:4H]); g = tanh([2H:3H])
 *   c = f⊙c + i⊙g;  h = o⊙tanh(c)
 */
tc_tensor_nd *tc_lstm_layer_forward(tc_lstm_layer *layer,
                                    const tc_tensor_nd *input) {
    if (!layer || !input || input->ndim != 3 ||
        input->shape[2] != tc_tensor_nd_shape(layer->weight_ih)[1])
        return NULL;
    const size_t seq = input->shape[0];
    const size_t batch = input->shape[1];
    const size_t input_features = input->shape[2];
    const size_t hidden = layer->hidden;

    const size_t h_shape[] = {batch, hidden};
    tc_tensor_nd *h = tc_tensor_nd_create(h_shape, 2, 1);
    tc_tensor_nd *c = tc_tensor_nd_create(h_shape, 2, 1);
    if (!h || !c) {
        tc_tensor_nd_free(h);
        tc_tensor_nd_free(c);
        return NULL;
    }

    /* W_ih^T [input, 4H] y W_hh^T [hidden, 4H]: se materializan una vez
     * por forward como ops de grafo (con su propio gradiente hacia W). */
    tc_tensor_nd *w_ih_t = tc_tensor_nd_transpose(layer->weight_ih);
    tc_tensor_nd *w_hh_t = tc_tensor_nd_transpose(layer->weight_hh);
    if (!w_ih_t || !w_hh_t) {
        tc_tensor_nd_free(w_ih_t);
        tc_tensor_nd_free(w_hh_t);
        tc_tensor_nd_free(h);
        tc_tensor_nd_free(c);
        return NULL;
    }

    tc_tensor_nd **outputs = (tc_tensor_nd **)malloc(seq * sizeof(*outputs));
    if (!outputs) {
        tc_tensor_nd_free(w_ih_t);
        tc_tensor_nd_free(w_hh_t);
        tc_tensor_nd_free(h);
        tc_tensor_nd_free(c);
        return NULL;
    }

    const size_t x2d_shape[] = {batch, input_features};
    int ok = 1;
    for (size_t t = 0; ok && t < seq; ++t) {
        tc_tensor_nd *x_t = tc_tensor_nd_slice_axis(input, 0, t, 1);
        tc_tensor_nd *x2d = x_t ? tc_tensor_nd_reshape_to(x_t, x2d_shape, 2)
                                : NULL;
        tc_tensor_nd_free(x_t);
        tc_tensor_nd *gx = x2d ? tc_tensor_nd_matmul(x2d, w_ih_t) : NULL;
        tc_tensor_nd_free(x2d);
        tc_tensor_nd *gh = tc_tensor_nd_matmul(h, w_hh_t);
        tc_tensor_nd *sum1 = gx && gh ? tc_tensor_nd_add(gx, gh) : NULL;
        tc_tensor_nd_free(gx);
        tc_tensor_nd_free(gh);
        tc_tensor_nd *gates = sum1 ? tc_tensor_nd_add(sum1, layer->bias)
                                   : NULL;
        tc_tensor_nd_free(sum1);
        if (!gates) { ok = 0; break; }

        /* Cada slice se materializa en variable local y se libera tras
         * crear su activación: la referencia de creación debe soltarse
         * explícitamente o el nodo queda huérfano con refcount 1. */
        tc_tensor_nd *s_i = tc_tensor_nd_slice_axis(gates, 1, 0, hidden);
        tc_tensor_nd *s_f = tc_tensor_nd_slice_axis(gates, 1, hidden, hidden);
        tc_tensor_nd *s_g = tc_tensor_nd_slice_axis(gates, 1, 2 * hidden,
                                                    hidden);
        tc_tensor_nd *s_o = tc_tensor_nd_slice_axis(gates, 1, 3 * hidden,
                                                    hidden);
        tc_tensor_nd *i_g = s_i ? tc_tensor_nd_sigmoid(s_i) : NULL;
        tc_tensor_nd *f_g = s_f ? tc_tensor_nd_sigmoid(s_f) : NULL;
        tc_tensor_nd *g_g = s_g ? tc_tensor_nd_tanh(s_g) : NULL;
        tc_tensor_nd *o_g = s_o ? tc_tensor_nd_sigmoid(s_o) : NULL;
        tc_tensor_nd_free(s_i);
        tc_tensor_nd_free(s_f);
        tc_tensor_nd_free(s_g);
        tc_tensor_nd_free(s_o);
        tc_tensor_nd_free(gates);
        if (!i_g || !f_g || !g_g || !o_g) {
            tc_tensor_nd_free(i_g);
            tc_tensor_nd_free(f_g);
            tc_tensor_nd_free(g_g);
            tc_tensor_nd_free(o_g);
            ok = 0;
            break;
        }

        /* c = f⊙c_prev + i⊙g */
        tc_tensor_nd *fc = tc_tensor_nd_mul(f_g, c);
        tc_tensor_nd *ig = tc_tensor_nd_mul(i_g, g_g);
        tc_tensor_nd_free(i_g);
        tc_tensor_nd_free(f_g);
        tc_tensor_nd_free(g_g);
        tc_tensor_nd *c_new = fc && ig ? tc_tensor_nd_add(fc, ig) : NULL;
        tc_tensor_nd_free(fc);
        tc_tensor_nd_free(ig);
        /* h = o⊙tanh(c) */
        tc_tensor_nd *tanh_c = c_new ? tc_tensor_nd_tanh(c_new) : NULL;
        tc_tensor_nd *h_new = o_g && tanh_c ? tc_tensor_nd_mul(o_g, tanh_c)
                                            : NULL;
        tc_tensor_nd_free(o_g);
        tc_tensor_nd_free(tanh_c);
        if (!c_new || !h_new) {
            tc_tensor_nd_free(c_new);
            tc_tensor_nd_free(h_new);
            ok = 0;
            break;
        }
        tc_tensor_nd_free(c);
        c = c_new;
        tc_tensor_nd_free(h);
        h = h_new;
        /* Vista [1, batch, hidden] de este paso para el concat final:
         * concat en el eje 0 de seq piezas produce [seq, batch, hidden]. */
        const size_t step_shape[] = {1, batch, hidden};
        outputs[t] = tc_tensor_nd_reshape_to(h, step_shape, 3);
        if (!outputs[t]) ok = 0;
    }

    tc_tensor_nd *result = NULL;
    if (ok) result = tc_tensor_nd_concat(
        (const tc_tensor_nd *const *)outputs, seq, 0);
    for (size_t t = 0; t < seq; ++t)
        if (outputs[t]) tc_tensor_nd_free(outputs[t]);
    free(outputs);
    tc_tensor_nd_free(w_ih_t);
    tc_tensor_nd_free(w_hh_t);
    tc_tensor_nd_free(h);
    tc_tensor_nd_free(c);
    return result;
}

tc_tensor_nd *tc_lstm_layer_weight_ih(tc_lstm_layer *layer) {
    return layer ? layer->weight_ih : NULL;
}

tc_tensor_nd *tc_lstm_layer_weight_hh(tc_lstm_layer *layer) {
    return layer ? layer->weight_hh : NULL;
}

tc_tensor_nd *tc_lstm_layer_bias(tc_lstm_layer *layer) {
    return layer ? layer->bias : NULL;
}
