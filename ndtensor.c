#include <tensorc/ndtensor.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    TC_ND_OP_LEAF = 0,
    TC_ND_OP_ADD,
    TC_ND_OP_MUL,
    TC_ND_OP_MATMUL,
    TC_ND_OP_RELU,
    TC_ND_OP_SUM,
    TC_ND_OP_MSE
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
    size_t parent_count;
    tc_tensor_nd *parents[2];
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
                                      tc_nd_op op, tc_tensor_nd *a,
                                      tc_tensor_nd *b) {
    const int requires_grad = (a && a->requires_grad) || (b && b->requires_grad);
    tc_tensor_nd *result = tensor_alloc(shape, ndim, requires_grad, 1);
    if (!result) return NULL;
    result->op = op;
    result->parent_count = b ? 2 : 1;
    result->parents[0] = a;
    result->parents[1] = b;
    tc_tensor_nd_retain(a);
    if (b) tc_tensor_nd_retain(b);
    return result;
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
    tc_tensor_nd *result = operation_result(shape, ndim, TC_ND_OP_ADD,
                                            (tc_tensor_nd *)a, (tc_tensor_nd *)b);
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
    tc_tensor_nd *result = operation_result(shape, ndim, TC_ND_OP_MUL,
                                            (tc_tensor_nd *)a, (tc_tensor_nd *)b);
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
    tc_tensor_nd *result = operation_result(shape, 2, TC_ND_OP_MATMUL,
                                            (tc_tensor_nd *)a, (tc_tensor_nd *)b);
    if (!result) return NULL;
    for (size_t i = 0; i < a->shape[0]; ++i)
        for (size_t k = 0; k < a->shape[1]; ++k)
            for (size_t j = 0; j < b->shape[1]; ++j)
                result->data[i * b->shape[1] + j] +=
                    a->data[i * a->shape[1] + k] * b->data[k * b->shape[1] + j];
    return result;
}

tc_tensor_nd *tc_tensor_nd_relu(const tc_tensor_nd *input) {
    if (!input) return NULL;
    tc_tensor_nd *result = operation_result(input->shape, input->ndim, TC_ND_OP_RELU,
                                            (tc_tensor_nd *)input, NULL);
    if (!result) return NULL;
    for (size_t i = 0; i < input->size; ++i)
        result->data[i] = input->data[i] > 0.0f ? input->data[i] : 0.0f;
    return result;
}

tc_tensor_nd *tc_tensor_nd_sum(const tc_tensor_nd *input) {
    if (!input) return NULL;
    const size_t shape[] = {1};
    tc_tensor_nd *result = operation_result(shape, 1, TC_ND_OP_SUM,
                                            (tc_tensor_nd *)input, NULL);
    if (!result) return NULL;
    for (size_t i = 0; i < input->size; ++i) result->data[0] += input->data[i];
    return result;
}

tc_tensor_nd *tc_tensor_nd_mse_loss(const tc_tensor_nd *prediction,
                                    const tc_tensor_nd *target) {
    if (!same_shape(prediction, target)) return NULL;
    const size_t shape[] = {1};
    tc_tensor_nd *result = operation_result(shape, 1, TC_ND_OP_MSE,
                                            (tc_tensor_nd *)prediction,
                                            (tc_tensor_nd *)target);
    if (!result) return NULL;
    for (size_t i = 0; i < prediction->size; ++i) {
        const float error = prediction->data[i] - target->data[i];
        result->data[0] += error * error;
    }
    result->data[0] /= (float)prediction->size;
    return result;
}

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
    tc_tensor_nd *a = tensor->parents[0];
    tc_tensor_nd *b = tensor->parents[1];
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
        case TC_ND_OP_RELU:
            for (size_t i = 0; i < a->size; ++i)
                add_grad(a, i, a->data[i] > 0.0f ? tensor->grad[i] : 0.0f);
            break;
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
