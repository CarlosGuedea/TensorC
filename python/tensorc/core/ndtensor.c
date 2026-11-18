#include <tensorc/ndtensor.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef TENSORC_HAS_ACCELERATE
#include <limits.h>
#include <Accelerate/Accelerate.h>
#endif

#ifdef TENSORC_HAS_CUDA
#include <limits.h>
#include "tensorc_cuda_internal.h"
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
    TC_ND_OP_AVGPOOL2D,
    TC_ND_OP_SOFTMAX,
    TC_ND_OP_CROSS_ENTROPY,
    TC_ND_OP_SIGMOID,
    TC_ND_OP_TANH,
    TC_ND_OP_SLICE,
    TC_ND_OP_CONCAT,
    TC_ND_OP_LAYERNORM,
    TC_ND_OP_TRANSPOSE,
    TC_ND_OP_ATTENTION,
    TC_ND_OP_PERMUTE
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
    /* Permutación de ejes para TC_ND_OP_PERMUTE (perm[i] = eje de la
     * entrada que va a la posición i de la salida). */
    size_t perm[8];
#ifdef TENSORC_HAS_CUDA
    /* Copias perezosas en GPU con flags de validez por copia. Invariante:
     * al menos una copia (host o device) es válida en todo momento. Las
     * rutas CUDA escriben en device y marcan la copia host como obsoleta;
     * los accesores y los caminos genéricos sincronizan bajo demanda. */
    float *dev_data;
    float *dev_grad;
    unsigned char host_data_valid;
    unsigned char dev_data_valid;
    unsigned char host_grad_valid;
    unsigned char dev_grad_valid;
    /* tc_tensor_nd_data() entrega un puntero host mutable: el llamador
     * puede seguir escribiendo a través de él sin que lo veamos. Mientras
     * esté expuesto, toda subida a device relee el host aunque la copia
     * de device parezca válida. */
    unsigned char host_data_exposed;
#endif
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
#ifdef TENSORC_HAS_CUDA
    /* calloc deja los punteros de device en NULL y sus flags en 0. */
    tensor->host_data_valid = 1;
    tensor->host_grad_valid = 1;
#endif
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
#ifdef TENSORC_HAS_CUDA
    tc__cuda_free(tensor->dev_data, tensor->size);
    tc__cuda_free(tensor->dev_grad, tensor->size);
#endif
    free(tensor);
}

#ifdef TENSORC_HAS_CUDA
/*
 * Sincronización perezosa host↔device. Convención:
 *  - nd_cuda_host_*: garantiza que la copia host está al día (baja de
 *    device solo si la marca host está obsoleta).
 *  - nd_cuda_dev_*:  garantiza buffer de device con contenido al día
 *    (sube de host solo si la marca device está obsoleta); 0 en fallo.
 *  - nd_cuda_produced_*: marca que un kernel dejó la copia device al día.
 * Cualquier fallo devuelve 0 y el llamador cae al camino genérico con el
 * estado consistente (las copias host de las entradas nunca se invalidan
 * antes de intentar la ruta CUDA).
 */
static void nd_cuda_host_data(const tc_tensor_nd *tensor) {
    tc_tensor_nd *t = (tc_tensor_nd *)tensor;
    if (!t || t->host_data_valid) return;
    if (t->dev_data &&
        tc__cuda_copy_d2h(t->data, t->dev_data, t->size)) {
        t->host_data_valid = 1;
        return;
    }
    /* Fallo de copia = dispositivo muerto: la copia host quedaría con
     * basura y cualquier resultado posterior sería incorrecto en
     * silencio. Se aborta con diagnóstico (misma política que en el
     * backward de matmul). */
    fprintf(stderr, "TensorC CUDA: no se pudo bajar data al host; "
                    "el estado del dispositivo es irrecuperable.\n");
    abort();
}

static void nd_cuda_host_grad(const tc_tensor_nd *tensor) {
    tc_tensor_nd *t = (tc_tensor_nd *)tensor;
    if (!t || !t->grad || t->host_grad_valid) return;
    if (t->dev_grad &&
        tc__cuda_copy_d2h(t->grad, t->dev_grad, t->size)) {
        t->host_grad_valid = 1;
        return;
    }
    fprintf(stderr, "TensorC CUDA: no se pudo bajar grad al host; "
                    "el estado del dispositivo es irrecuperable.\n");
    abort();
}

static int nd_cuda_dev_data(tc_tensor_nd *tensor) {
    if (tensor->dev_data_valid && !tensor->host_data_exposed) return 1;
    if (!tensor->dev_data) {
        tensor->dev_data = tc__cuda_alloc(tensor->size);
        if (!tensor->dev_data) return 0;
    }
    if (tensor->host_data_valid &&
        !tc__cuda_copy_h2d(tensor->dev_data, tensor->data, tensor->size))
        return 0;
    tensor->dev_data_valid = 1;
    return 1;
}

static int nd_cuda_dev_grad(tc_tensor_nd *tensor) {
    if (!tensor->grad) return 0;
    if (tensor->dev_grad_valid) return 1;
    if (!tensor->dev_grad) {
        tensor->dev_grad = tc__cuda_alloc(tensor->size);
        if (!tensor->dev_grad) return 0;
    }
    if (tensor->host_grad_valid &&
        !tc__cuda_copy_h2d(tensor->dev_grad, tensor->grad, tensor->size))
        return 0;
    tensor->dev_grad_valid = 1;
    return 1;
}

/* Solo reserva el buffer de salida: el kernel sobrescribe cada elemento. */
static int nd_cuda_dev_alloc_data(tc_tensor_nd *tensor) {
    if (!tensor->dev_data) {
        tensor->dev_data = tc__cuda_alloc(tensor->size);
        if (!tensor->dev_data) return 0;
    }
    return 1;
}

static void nd_cuda_produced_data(tc_tensor_nd *tensor) {
    tensor->dev_data_valid = 1;
    tensor->host_data_valid = 0;
}

static void nd_cuda_produced_grad(tc_tensor_nd *tensor) {
    tensor->dev_grad_valid = 1;
    tensor->host_grad_valid = 0;
}

/* Descriptor de broadcasting para los kernels (límite TC_CUDA_MAX_DIMS). */
static int nd_cuda_broadcast_desc(const tc_tensor_nd *a, const tc_tensor_nd *b,
                                  const tc_tensor_nd *out,
                                  tc_cuda_broadcast_desc *desc) {
    if (a->ndim > TC_CUDA_MAX_DIMS || b->ndim > TC_CUDA_MAX_DIMS ||
        out->ndim > TC_CUDA_MAX_DIMS)
        return 0;
    memset(desc, 0, sizeof(*desc));
    desc->ndim = (int)out->ndim;
    desc->a_ndim = (int)a->ndim;
    desc->b_ndim = (int)b->ndim;
    for (size_t i = 0; i < out->ndim; ++i) {
        desc->out_shape[i] = out->shape[i];
        desc->out_strides[i] = out->strides[i];
    }
    for (size_t i = 0; i < a->ndim; ++i) {
        desc->a_shape[i] = a->shape[i];
        desc->a_strides[i] = a->strides[i];
    }
    for (size_t i = 0; i < b->ndim; ++i) {
        desc->b_shape[i] = b->shape[i];
        desc->b_strides[i] = b->strides[i];
    }
    return 1;
}

/* Descriptor de permutación para los kernels (transpose se expresa como
 * axes = {1, 0}). */
static int nd_cuda_permute_desc(const tc_tensor_nd *in,
                                const tc_tensor_nd *out, const size_t *axes,
                                tc_cuda_permute_desc *desc) {
    if (in->ndim > TC_CUDA_MAX_DIMS || out->ndim > TC_CUDA_MAX_DIMS)
        return 0;
    memset(desc, 0, sizeof(*desc));
    desc->ndim = (int)in->ndim;
    for (size_t i = 0; i < in->ndim; ++i) {
        desc->in_shape[i] = in->shape[i];
        desc->in_strides[i] = in->strides[i];
        desc->out_strides[i] = out->strides[i];
        desc->axes[i] = axes[i];
    }
    return 1;
}

/*
 * Regla de residencia: una op va a GPU si supera el umbral de tamaño o si
 * su entrada ya reside en device (caer a CPU pagaría el par de copias
 * D2H+H2D por puro traslado y rompería las cadenas de ops pequeñas, p. ej.
 * el bloque Transformer). En el backward la residencia se mide sobre el
 * gradiente del nodo de salida.
 */
#define ND_CUDA_RESIDENT(t) \
    ((t)->size >= tc__cuda_min_elems() || (t)->dev_data_valid)
#define ND_CUDA_RESIDENT_GRAD(t) \
    ((t)->size >= tc__cuda_min_elems() || (t)->dev_grad_valid)

#define ND_SYNC_DATA(t) nd_cuda_host_data(t)
#else
#define ND_SYNC_DATA(t) ((void)0)
#endif

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
    if (!tensor) return NULL;
#ifdef TENSORC_HAS_CUDA
    nd_cuda_host_data(tensor);
    /* Puntero mutable: el llamador puede escribir ahora y DESPUÉS a través
     * del mismo puntero. La copia de device queda obsoleta y marcada como
     * expuesta: cada subida posterior relee el host por seguridad. */
    tensor->dev_data_valid = 0;
    tensor->host_data_exposed = 1;
#endif
    return tensor->data;
}

const float *tc_tensor_nd_const_data(const tc_tensor_nd *tensor) {
    if (!tensor) return NULL;
#ifdef TENSORC_HAS_CUDA
    nd_cuda_host_data(tensor);
#endif
    return tensor->data;
}

const float *tc_tensor_nd_grad(const tc_tensor_nd *tensor) {
    if (!tensor) return NULL;
#ifdef TENSORC_HAS_CUDA
    nd_cuda_host_grad(tensor);
#endif
    return tensor->grad;
}

int tc_tensor_nd_requires_grad(const tc_tensor_nd *tensor) {
    return tensor ? tensor->requires_grad : 0;
}

void tc_tensor_nd_zero_grad(tc_tensor_nd *tensor) {
    if (tensor && tensor->grad) {
#ifdef TENSORC_HAS_CUDA
        /* Si el gradiente ya tiene buffer de device, se pone a cero ahí
         * mismo y la copia host queda obsoleta (evita subir ceros). */
        if (tensor->dev_grad && tc_cuda_available() &&
            tc__cuda_zero_fill(tensor->dev_grad, tensor->size)) {
            tensor->dev_grad_valid = 1;
            tensor->host_grad_valid = 0;
            return;
        }
#endif
        memset(tensor->grad, 0, tensor->size * sizeof(float));
#ifdef TENSORC_HAS_CUDA
        tensor->host_grad_valid = 1;
        tensor->dev_grad_valid = 0;
#endif
    }
}

int tc_tensor_nd_copy_(tc_tensor_nd *dst, const tc_tensor_nd *src) {
    if (!dst || !src || dst->ndim != src->ndim) return 0;
    for (size_t i = 0; i < dst->ndim; ++i)
        if (dst->shape[i] != src->shape[i]) return 0;
    ND_SYNC_DATA(src);
    memcpy(dst->data, src->data, src->size * sizeof(float));
#ifdef TENSORC_HAS_CUDA
    dst->host_data_valid = 1;
    dst->dev_data_valid = 0;
#endif
    return 1;
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

/* Devuelve 1 si `full` tiene exactamente la forma de `out` y `vec` es el
 * vector fila [1, out->shape[1]] que se difunde sobre las filas — el
 * patrón sesgo de cada capa. Evita la aritmética de strides por elemento. */
static int is_row_broadcast(const tc_tensor_nd *full, const tc_tensor_nd *vec,
                            const tc_tensor_nd *out, size_t *cols) {
    if (full->ndim != 2 || out->ndim != 2 || vec->ndim != 2 ||
        full->shape[0] != out->shape[0] ||
        full->shape[1] != out->shape[1] || vec->shape[0] != 1 ||
        vec->shape[1] != out->shape[1]) return 0;
    *cols = out->shape[1];
    return 1;
}

/* Suma con difusión de filas: out[i,j] = a[i,j] (+|*) v[j]. */
static void row_broadcast_binary(const tc_tensor_nd *a, const tc_tensor_nd *v,
                                 tc_tensor_nd *result, int multiply) {
    const size_t rows = result->shape[0], cols = result->shape[1];
    if (multiply) {
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                result->data[i * cols + j] =
                    a->data[i * cols + j] * v->data[j];
    } else {
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                result->data[i * cols + j] =
                    a->data[i * cols + j] + v->data[j];
    }
}

tc_tensor_nd *tc_tensor_nd_add(const tc_tensor_nd *a, const tc_tensor_nd *b) {
    size_t *shape = NULL;
    size_t ndim = 0;
    if (!broadcast_shape(a, b, &shape, &ndim)) return NULL;
    tc_tensor_nd *result = binary_result(shape, ndim, TC_ND_OP_ADD,
                                            a, b);
    free(shape);
    if (!result) return NULL;
#ifdef TENSORC_HAS_CUDA
    if (tc_cuda_available() &&
        (ND_CUDA_RESIDENT(a) || ND_CUDA_RESIDENT(b))) {
        tc_cuda_broadcast_desc desc;
        if (nd_cuda_broadcast_desc(a, b, result, &desc) &&
            nd_cuda_dev_data((tc_tensor_nd *)a) &&
            nd_cuda_dev_data((tc_tensor_nd *)b) &&
            nd_cuda_dev_alloc_data(result) &&
            tc__cuda_add_forward(a->dev_data, b->dev_data, result->dev_data,
                                 result->size, &desc)) {
            nd_cuda_produced_data(result);
            return result;
        }
    }
#endif
    ND_SYNC_DATA(a);
    ND_SYNC_DATA(b);
    size_t cols = 0;
    if (is_row_broadcast(a, b, result, &cols)) {
        row_broadcast_binary(a, b, result, 0);
        return result;
    }
    if (is_row_broadcast(b, a, result, &cols)) {
        row_broadcast_binary(b, a, result, 0);
        return result;
    }
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
#ifdef TENSORC_HAS_CUDA
    if (tc_cuda_available() &&
        (ND_CUDA_RESIDENT(a) || ND_CUDA_RESIDENT(b))) {
        tc_cuda_broadcast_desc desc;
        if (nd_cuda_broadcast_desc(a, b, result, &desc) &&
            nd_cuda_dev_data((tc_tensor_nd *)a) &&
            nd_cuda_dev_data((tc_tensor_nd *)b) &&
            nd_cuda_dev_alloc_data(result) &&
            tc__cuda_mul_forward(a->dev_data, b->dev_data, result->dev_data,
                                 result->size, &desc)) {
            nd_cuda_produced_data(result);
            return result;
        }
    }
#endif
    ND_SYNC_DATA(a);
    ND_SYNC_DATA(b);
    size_t cols = 0;
    if (is_row_broadcast(a, b, result, &cols)) {
        row_broadcast_binary(a, b, result, 1);
        return result;
    }
    if (is_row_broadcast(b, a, result, &cols)) {
        row_broadcast_binary(b, a, result, 1);
        return result;
    }
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
#ifdef TENSORC_HAS_CUDA
    /* Ruta rápida GPU: cuBLAS con el mismo criterio de tamaño que
     * Accelerate. El resultado queda residente en device. */
    {
        const size_t m = a->shape[0], k = a->shape[1], n = b->shape[1];
        if (tc_cuda_available() &&
            m <= (size_t)INT_MAX && k <= (size_t)INT_MAX &&
            n <= (size_t)INT_MAX && m > 0 && k > 0 && n > 0 &&
            m <= ((size_t)-1) / n && k <= ((size_t)-1) / n &&
            m * n * k >= tc__cuda_min_gemm_work() &&
            nd_cuda_dev_data((tc_tensor_nd *)a) &&
            nd_cuda_dev_data((tc_tensor_nd *)b) &&
            nd_cuda_dev_alloc_data(result) &&
            tc__cuda_matmul_nn(a->dev_data, b->dev_data, result->dev_data,
                               m, k, n)) {
            nd_cuda_produced_data(result);
            return result;
        }
    }
#endif
    ND_SYNC_DATA(a);
    ND_SYNC_DATA(b);
#ifdef TENSORC_HAS_ACCELERATE
    /* Ruta rápida: GEMM de Accelerate para tamaños donde el triple loop
     * escalar ya se nota. El producto m*n*k acota el trabajo. */
    const size_t m = a->shape[0], k = a->shape[1], n = b->shape[1];
    if (m <= (size_t)INT_MAX && k <= (size_t)INT_MAX &&
        n <= (size_t)INT_MAX && m > 0 && k > 0 && n > 0 &&
        m <= ((size_t)-1) / n && k <= ((size_t)-1) / n &&
        m * n * k >= ((size_t)1 << 12)) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    (int)m, (int)n, (int)k,
                    1.0f, a->data, (int)k, b->data, (int)n,
                    0.0f, result->data, (int)n);
        return result;
    }
#endif
    for (size_t i = 0; i < a->shape[0]; ++i)
        for (size_t k2 = 0; k2 < a->shape[1]; ++k2)
            for (size_t j = 0; j < b->shape[1]; ++j)
                result->data[i * b->shape[1] + j] +=
                    a->data[i * a->shape[1] + k2] * b->data[k2 * b->shape[1] + j];
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
    ND_SYNC_DATA(input);
    ND_SYNC_DATA(kernel);
    if (conv2d_accelerate_forward(input, kernel, options, result))
        return result;
#endif
    ND_SYNC_DATA(input);
    ND_SYNC_DATA(kernel);
    conv2d_forward_naive(input, kernel, options, result);
    return result;
}

tc_tensor_nd *tc_tensor_nd_relu(const tc_tensor_nd *input) {
    if (!input) return NULL;
    tc_tensor_nd *result = unary_result(input->shape, input->ndim, TC_ND_OP_RELU,
                                            input);
    if (!result) return NULL;
#ifdef TENSORC_HAS_CUDA
    if (tc_cuda_available() && ND_CUDA_RESIDENT(input) &&
        nd_cuda_dev_data((tc_tensor_nd *)input) &&
        nd_cuda_dev_alloc_data(result) &&
        tc__cuda_relu_forward(input->dev_data, result->dev_data,
                              input->size)) {
        nd_cuda_produced_data(result);
        return result;
    }
#endif
    ND_SYNC_DATA(input);
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
#ifdef TENSORC_HAS_CUDA
    if (tc_cuda_available() && ND_CUDA_RESIDENT(input) &&
        nd_cuda_dev_data((tc_tensor_nd *)input) &&
        nd_cuda_dev_alloc_data(result) &&
        tc__cuda_sigmoid_forward(input->dev_data, result->dev_data,
                                 input->size)) {
        nd_cuda_produced_data(result);
        return result;
    }
#endif
    ND_SYNC_DATA(input);
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
#ifdef TENSORC_HAS_CUDA
    if (tc_cuda_available() && ND_CUDA_RESIDENT(input) &&
        nd_cuda_dev_data((tc_tensor_nd *)input) &&
        nd_cuda_dev_alloc_data(result) &&
        tc__cuda_tanh_forward(input->dev_data, result->dev_data,
                              input->size)) {
        nd_cuda_produced_data(result);
        return result;
    }
#endif
    ND_SYNC_DATA(input);
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
#ifdef TENSORC_HAS_CUDA
    if (tc_cuda_available() && ND_CUDA_RESIDENT(input)) {
        const size_t classes_ = input->shape[input->ndim - 1];
        const size_t rows_ = input->size / classes_;
        if (nd_cuda_dev_data((tc_tensor_nd *)input) &&
            nd_cuda_dev_alloc_data(result) &&
            tc__cuda_softmax_forward(input->dev_data, result->dev_data,
                                     rows_, classes_)) {
            nd_cuda_produced_data(result);
            return result;
        }
    }
#endif
    ND_SYNC_DATA(input);
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
#ifdef TENSORC_HAS_CUDA
    /* Reducción en device: la loss nunca baja a host si no la leen. */
    if (tc_cuda_available() && ND_CUDA_RESIDENT(prediction) &&
        nd_cuda_dev_data((tc_tensor_nd *)prediction) &&
        nd_cuda_dev_data((tc_tensor_nd *)target) &&
        nd_cuda_dev_alloc_data(result) &&
        tc__cuda_cross_entropy_forward(prediction->dev_data,
                                       target->dev_data, result->dev_data,
                                       prediction->size, rows)) {
        nd_cuda_produced_data(result);
        return result;
    }
#endif
    ND_SYNC_DATA(prediction);
    ND_SYNC_DATA(target);
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
#ifdef TENSORC_HAS_CUDA
    /* Copia device→device: el reshape no reordena datos. */
    if (tc_cuda_available() && ND_CUDA_RESIDENT(input) &&
        nd_cuda_dev_data((tc_tensor_nd *)input) &&
        nd_cuda_dev_alloc_data(result) &&
        tc__cuda_copy_d2d(result->dev_data, input->dev_data, input->size)) {
        nd_cuda_produced_data(result);
        return result;
    }
#endif
    ND_SYNC_DATA(input);
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

    ND_SYNC_DATA(input);
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

tc_tensor_nd *tc_tensor_nd_avgpool2d(const tc_tensor_nd *input,
                                     size_t pool_h, size_t pool_w) {
    if (!input || input->ndim != 4 || pool_h == 0 || pool_w == 0 ||
        input->shape[2] % pool_h != 0 || input->shape[3] % pool_w != 0)
        return NULL;
    const size_t output_shape[] = {
        input->shape[0], input->shape[1],
        input->shape[2] / pool_h, input->shape[3] / pool_w
    };
    tc_tensor_nd *result = unary_result(output_shape, 4, TC_ND_OP_AVGPOOL2D,
                                            input);
    if (!result) return NULL;

    ND_SYNC_DATA(input);
    const size_t batch = input->shape[0];
    const size_t channels = input->shape[1];
    const size_t input_height = input->shape[2];
    const size_t input_width = input->shape[3];
    const size_t output_height = result->shape[2];
    const size_t output_width = result->shape[3];
    const float window = (float)(pool_h * pool_w);
    for (size_t n = 0; n < batch; ++n)
        for (size_t c = 0; c < channels; ++c)
            for (size_t oh = 0; oh < output_height; ++oh)
                for (size_t ow = 0; ow < output_width; ++ow) {
                    float sum = 0.0f;
                    const size_t row0 = oh * pool_h;
                    const size_t col0 = ow * pool_w;
                    for (size_t ph = 0; ph < pool_h; ++ph)
                        for (size_t pw = 0; pw < pool_w; ++pw)
                            sum += input->data[
                                (((n * channels + c) * input_height + row0 + ph) *
                                 input_width + col0 + pw)];
                    result->data[(((n * channels + c) * output_height + oh) *
                                  output_width + ow)] = sum / window;
                }
    return result;
}

tc_tensor_nd *tc_tensor_nd_sum(const tc_tensor_nd *input) {
    if (!input) return NULL;
    const size_t shape[] = {1};
    tc_tensor_nd *result = unary_result(shape, 1, TC_ND_OP_SUM,
                                            input);
    if (!result) return NULL;
    ND_SYNC_DATA(input);
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
    ND_SYNC_DATA(prediction);
    ND_SYNC_DATA(target);
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
    ND_SYNC_DATA(input);
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
        ND_SYNC_DATA(inputs[0]);
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
    for (size_t i = 0; i < count; ++i) ND_SYNC_DATA(inputs[i]);
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
#ifdef TENSORC_HAS_CUDA
    if (tc_cuda_available() && ND_CUDA_RESIDENT(input) &&
        nd_cuda_dev_data((tc_tensor_nd *)input) &&
        nd_cuda_dev_alloc_data(result) &&
        tc__cuda_layer_norm_forward(input->dev_data, result->dev_data,
                                    rows, width)) {
        nd_cuda_produced_data(result);
        return result;
    }
#endif
    ND_SYNC_DATA(input);
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
#ifdef TENSORC_HAS_CUDA
    /* Transpose 2D como permute de ejes {1, 0}. */
    if (tc_cuda_available() && ND_CUDA_RESIDENT(input)) {
        const size_t axes2[] = {1, 0};
        tc_cuda_permute_desc desc;
        if (nd_cuda_permute_desc(input, result, axes2, &desc) &&
            nd_cuda_dev_data((tc_tensor_nd *)input) &&
            nd_cuda_dev_alloc_data(result) &&
            tc__cuda_permute_forward(input->dev_data, result->dev_data,
                                     input->size, &desc)) {
            nd_cuda_produced_data(result);
            return result;
        }
    }
#endif
    ND_SYNC_DATA(input);
    for (size_t i = 0; i < input->shape[0]; ++i)
        for (size_t j = 0; j < input->shape[1]; ++j)
            result->data[j * input->shape[0] + i] = input->data[i * input->shape[1] + j];
    return result;
}

/*
 * Permutación general de ejes: la salida satisface
 * out[i0, i1, ...] = in[i_{axes[0]}, i_{axes[1]}, ...]. Los ejes deben ser
 * una permutación completa de 0..ndim-1 y ndim == rango de la entrada.
 */
tc_tensor_nd *tc_tensor_nd_permute(const tc_tensor_nd *input,
                                   const size_t *axes, size_t ndim) {
    if (!input || !axes || ndim == 0 || ndim != input->ndim || ndim > 8)
        return NULL;
    int seen[8] = {0};
    size_t shape[8];
    for (size_t i = 0; i < ndim; ++i) {
        if (axes[i] >= ndim || seen[axes[i]]) return NULL;
        seen[axes[i]] = 1;
        shape[i] = input->shape[axes[i]];
    }
    tc_tensor_nd *result = unary_result(shape, ndim, TC_ND_OP_PERMUTE, input);
    if (!result) return NULL;
    for (size_t i = 0; i < ndim; ++i) result->perm[i] = axes[i];
#ifdef TENSORC_HAS_CUDA
    if (tc_cuda_available() && ND_CUDA_RESIDENT(input)) {
        tc_cuda_permute_desc desc;
        if (nd_cuda_permute_desc(input, result, axes, &desc) &&
            nd_cuda_dev_data((tc_tensor_nd *)input) &&
            nd_cuda_dev_alloc_data(result) &&
            tc__cuda_permute_forward(input->dev_data, result->dev_data,
                                     input->size, &desc)) {
            nd_cuda_produced_data(result);
            return result;
        }
    }
#endif
    ND_SYNC_DATA(input);
    /* Recorrido lineal de la entrada: cada índice lineal se descompone en
     * índices multidimensionales y se reensambla en la posición de salida
     * usando la permutación. */
    size_t in_idx[8];
    for (size_t lin = 0; lin < input->size; ++lin) {
        for (size_t d = 0; d < ndim; ++d)
            in_idx[d] = (lin / input->strides[d]) % input->shape[d];
        size_t out_lin = 0;
        for (size_t k = 0; k < ndim; ++k)
            out_lin += in_idx[axes[k]] * result->strides[k];
        result->data[out_lin] = input->data[lin];
    }
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
#ifdef TENSORC_HAS_CUDA
    /* Ruta GPU: scores = scale·Q·Kᵀ y salida = P·V como GEMM batidas con
     * strides (cuBLAS) + softmax (causal opcional) en kernel propio. Va a
     * GPU si el volumen de scores supera el umbral o si las tres entradas
     * ya residen en device (típico dentro del bloque Transformer). */
    if (tc_cuda_available() &&
        (batch * q_seq * k_seq >= tc__cuda_min_elems() ||
         (query->dev_data_valid && key->dev_data_valid &&
          value->dev_data_valid))) {
        if (nd_cuda_dev_data((tc_tensor_nd *)query) &&
            nd_cuda_dev_data((tc_tensor_nd *)key) &&
            nd_cuda_dev_data((tc_tensor_nd *)value) &&
            nd_cuda_dev_alloc_data(result) &&
            tc__cuda_attention_forward(
                query->dev_data, key->dev_data, value->dev_data,
                result->dev_data, batch, q_seq, k_seq, depth, v_depth,
                result->attn_causal)) {
            nd_cuda_produced_data(result);
            return result;
        }
    }
#endif
    ND_SYNC_DATA(query);
    ND_SYNC_DATA(key);
    ND_SYNC_DATA(value);
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
    if (tensor && tensor->requires_grad && tensor->grad) {
#ifdef TENSORC_HAS_CUDA
        /* El backward genérico escribe en host: sincroniza e invalida la
         * copia de device. */
        nd_cuda_host_grad(tensor);
        tensor->dev_grad_valid = 0;
#endif
        tensor->grad[index] += value;
    }
}

#ifdef TENSORC_HAS_CUDA
/*
 * Dispatcher del backward en GPU. Devuelve 1 si procesó el nodo por
 * completo; 0 pide el camino genérico (ndtensor.c sincroniza a host antes
 * de caer). Las pruebas previas (disponibilidad, tamaños, buffers) se
 * hacen antes de lanzar trabajo; los kernels se verifican con sincronía
 * inmediata, de modo que cada op es todo-o-nada. En el backward de matmul
 * hay dos GEMM encadenados que acumulan: si el segundo fallara tras
 * completarse el primero el estado sería irrecuperable, así que se aborta
 * con diagnóstico (fallo de driver/hardware; la alternativa sería
 * devolver gradientes corruptos en silencio).
 */
static int nd_cuda_backward_node(tc_tensor_nd *tensor, tc_tensor_nd *a,
                                 tc_tensor_nd *b) {
    if (!tc_cuda_available() || !tensor->grad) return 0;
    switch (tensor->op) {
        case TC_ND_OP_MATMUL: {
            const size_t m = a->shape[0], k = a->shape[1], n = b->shape[1];
            if (m == 0 || k == 0 || n == 0 || m > (size_t)INT_MAX ||
                k > (size_t)INT_MAX || n > (size_t)INT_MAX ||
                m > ((size_t)-1) / n || k > ((size_t)-1) / n ||
                m * n * k < tc__cuda_min_gemm_work())
                return 0;
            const int want_a = a->requires_grad && a->grad;
            const int want_b = b->requires_grad && b->grad;
            if (!want_a && !want_b) return 1;
            if (!nd_cuda_dev_grad(tensor)) return 0;
            if (want_a &&
                (!nd_cuda_dev_data(b) || !nd_cuda_dev_grad(a)))
                return 0;
            if (want_b &&
                (!nd_cuda_dev_data(a) || !nd_cuda_dev_grad(b)))
                return 0;
            if (want_a) {
                if (!tc__cuda_matmul_nt(tensor->dev_grad, b->dev_data,
                                        a->dev_grad, m, k, n))
                    return 0;
                nd_cuda_produced_grad(a);
            }
            if (want_b) {
                if (!tc__cuda_matmul_tn(a->dev_data, tensor->dev_grad,
                                        b->dev_grad, m, k, n)) {
                    if (want_a) {
                        fprintf(stderr,
                                "TensorC CUDA: la segunda GEMM del backward "
                                "falló tras acumular dA; gradientes "
                                "irrecuperables.\n");
                        abort();
                    }
                    return 0;
                }
                nd_cuda_produced_grad(b);
            }
            return 1;
        }
        case TC_ND_OP_ADD: {
            if (tensor->size < tc__cuda_min_elems() &&
                !tensor->dev_grad_valid)
                return 0;
            tc_cuda_broadcast_desc desc;
            if (!nd_cuda_broadcast_desc(a, b, tensor, &desc)) return 0;
            const int want_a = a->requires_grad && a->grad;
            const int want_b = b->requires_grad && b->grad;
            if (!want_a && !want_b) return 1;
            if (!nd_cuda_dev_grad(tensor)) return 0;
            if (want_a && !nd_cuda_dev_grad(a)) return 0;
            if (want_b && !nd_cuda_dev_grad(b)) return 0;
            if (!tc__cuda_add_backward(tensor->dev_grad,
                                       want_a ? a->dev_grad : NULL,
                                       want_b ? b->dev_grad : NULL,
                                       tensor->size, &desc, want_a, want_b))
                return 0;
            if (want_a) nd_cuda_produced_grad(a);
            if (want_b) nd_cuda_produced_grad(b);
            return 1;
        }
        case TC_ND_OP_MUL: {
            if (tensor->size < tc__cuda_min_elems() &&
                !tensor->dev_grad_valid)
                return 0;
            tc_cuda_broadcast_desc desc;
            if (!nd_cuda_broadcast_desc(a, b, tensor, &desc)) return 0;
            const int want_a = a->requires_grad && a->grad;
            const int want_b = b->requires_grad && b->grad;
            if (!want_a && !want_b) return 1;
            if (!nd_cuda_dev_grad(tensor)) return 0;
            if (!nd_cuda_dev_data(a) || !nd_cuda_dev_data(b)) return 0;
            if (want_a && !nd_cuda_dev_grad(a)) return 0;
            if (want_b && !nd_cuda_dev_grad(b)) return 0;
            if (!tc__cuda_mul_backward(tensor->dev_grad, a->dev_data,
                                       b->dev_data,
                                       want_a ? a->dev_grad : NULL,
                                       want_b ? b->dev_grad : NULL,
                                       tensor->size, &desc, want_a, want_b))
                return 0;
            if (want_a) nd_cuda_produced_grad(a);
            if (want_b) nd_cuda_produced_grad(b);
            return 1;
        }
        case TC_ND_OP_RELU: {
            if ((a->size < tc__cuda_min_elems() &&
                 !tensor->dev_grad_valid) ||
                !a->requires_grad || !a->grad)
                return 0;
            if (!nd_cuda_dev_grad(tensor) || !nd_cuda_dev_data(a) ||
                !nd_cuda_dev_grad(a))
                return 0;
            if (!tc__cuda_relu_backward(a->dev_data, tensor->dev_grad,
                                        a->dev_grad, a->size))
                return 0;
            nd_cuda_produced_grad(a);
            return 1;
        }
        case TC_ND_OP_SIGMOID: {
            if ((a->size < tc__cuda_min_elems() &&
                 !tensor->dev_grad_valid) ||
                !a->requires_grad || !a->grad)
                return 0;
            /* El backward usa la salida y = sigmoid(x), que es tensor. */
            if (!nd_cuda_dev_grad(tensor) || !nd_cuda_dev_data(tensor) ||
                !nd_cuda_dev_grad(a))
                return 0;
            if (!tc__cuda_sigmoid_backward(tensor->dev_data,
                                           tensor->dev_grad, a->dev_grad,
                                           a->size))
                return 0;
            nd_cuda_produced_grad(a);
            return 1;
        }
        case TC_ND_OP_TANH: {
            if ((a->size < tc__cuda_min_elems() &&
                 !tensor->dev_grad_valid) ||
                !a->requires_grad || !a->grad)
                return 0;
            if (!nd_cuda_dev_grad(tensor) || !nd_cuda_dev_data(tensor) ||
                !nd_cuda_dev_grad(a))
                return 0;
            if (!tc__cuda_tanh_backward(tensor->dev_data, tensor->dev_grad,
                                        a->dev_grad, a->size))
                return 0;
            nd_cuda_produced_grad(a);
            return 1;
        }
        case TC_ND_OP_SOFTMAX: {
            if ((a->size < tc__cuda_min_elems() &&
                 !tensor->dev_grad_valid) ||
                !a->requires_grad || !a->grad)
                return 0;
            const size_t classes = a->shape[a->ndim - 1];
            const size_t rows = a->size / classes;
            if (!nd_cuda_dev_grad(tensor) || !nd_cuda_dev_data(tensor) ||
                !nd_cuda_dev_grad(a))
                return 0;
            if (!tc__cuda_softmax_backward(tensor->dev_data,
                                           tensor->dev_grad, a->dev_grad,
                                           rows, classes))
                return 0;
            nd_cuda_produced_grad(a);
            return 1;
        }
        case TC_ND_OP_CROSS_ENTROPY: {
            const size_t classes = a->shape[a->ndim - 1];
            const size_t rows = a->size / classes;
            if (a->size < tc__cuda_min_elems() && !tensor->dev_grad_valid)
                return 0;
            const int want_a = a->requires_grad && a->grad;
            const int want_b = b->requires_grad && b->grad;
            if (!want_a && !want_b) return 1;
            if (!nd_cuda_dev_data(a) || !nd_cuda_dev_data(b) ||
                !nd_cuda_dev_grad(tensor))
                return 0;
            if (want_a && !nd_cuda_dev_grad(a)) return 0;
            if (want_b && !nd_cuda_dev_grad(b)) return 0;
            if (!tc__cuda_cross_entropy_backward(
                    a->dev_data, b->dev_data, tensor->dev_grad,
                    want_a ? a->dev_grad : NULL,
                    want_b ? b->dev_grad : NULL,
                    a->size, rows, want_a, want_b))
                return 0;
            if (want_a) nd_cuda_produced_grad(a);
            if (want_b) nd_cuda_produced_grad(b);
            return 1;
        }
        case TC_ND_OP_LAYERNORM: {
            if (!a->requires_grad || !a->grad) return 1;
            if (a->size < tc__cuda_min_elems() && !tensor->dev_grad_valid)
                return 0;
            const size_t width = a->shape[a->ndim - 1];
            const size_t rows = a->size / width;
            if (!nd_cuda_dev_grad(tensor) || !nd_cuda_dev_data(a) ||
                !nd_cuda_dev_grad(a))
                return 0;
            if (!tc__cuda_layer_norm_backward(a->dev_data, tensor->dev_grad,
                                              a->dev_grad, rows, width))
                return 0;
            nd_cuda_produced_grad(a);
            return 1;
        }
        case TC_ND_OP_ATTENTION: {
            tc_tensor_nd *q = tensor->parents[0];
            tc_tensor_nd *k = tensor->parents[1];
            tc_tensor_nd *v = tensor->parents[2];
            const size_t batch = q->shape[0];
            const size_t q_seq = q->shape[1];
            const size_t k_seq = k->shape[1];
            const size_t depth = q->shape[2];
            const size_t v_depth = v->shape[2];
            if (batch * q_seq * k_seq < tc__cuda_min_elems() &&
                !tensor->dev_grad_valid)
                return 0;
            const int want_q = q->requires_grad && q->grad;
            const int want_k = k->requires_grad && k->grad;
            const int want_v = v->requires_grad && v->grad;
            if (!want_q && !want_k && !want_v) return 1;
            if (!nd_cuda_dev_grad(tensor) || !nd_cuda_dev_data(q) ||
                !nd_cuda_dev_data(k) || !nd_cuda_dev_data(v))
                return 0;
            if (want_q && !nd_cuda_dev_grad(q)) return 0;
            if (want_k && !nd_cuda_dev_grad(k)) return 0;
            if (want_v && !nd_cuda_dev_grad(v)) return 0;
            if (!tc__cuda_attention_backward(
                    q->dev_data, k->dev_data, v->dev_data, tensor->dev_grad,
                    want_q ? q->dev_grad : NULL,
                    want_k ? k->dev_grad : NULL,
                    want_v ? v->dev_grad : NULL,
                    batch, q_seq, k_seq, depth, v_depth,
                    tensor->attn_causal))
                return 0;
            if (want_q) nd_cuda_produced_grad(q);
            if (want_k) nd_cuda_produced_grad(k);
            if (want_v) nd_cuda_produced_grad(v);
            return 1;
        }
        case TC_ND_OP_TRANSPOSE:
        case TC_ND_OP_PERMUTE: {
            if (!a->requires_grad || !a->grad) return 1;
            if (a->size < tc__cuda_min_elems() && !tensor->dev_grad_valid)
                return 0;
            const size_t transpose_axes[] = {1, 0};
            const size_t *axes = tensor->op == TC_ND_OP_PERMUTE
                                     ? tensor->perm
                                     : transpose_axes;
            tc_cuda_permute_desc desc;
            if (!nd_cuda_permute_desc(a, tensor, axes, &desc)) return 0;
            if (!nd_cuda_dev_grad(tensor) || !nd_cuda_dev_grad(a)) return 0;
            if (!tc__cuda_permute_backward(tensor->dev_grad, a->dev_grad,
                                           a->size, &desc))
                return 0;
            nd_cuda_produced_grad(a);
            return 1;
        }
        case TC_ND_OP_RESHAPE: {
            if (!a->requires_grad || !a->grad) return 1;
            if (a->size < tc__cuda_min_elems() && !tensor->dev_grad_valid)
                return 0;
            if (!nd_cuda_dev_grad(tensor) || !nd_cuda_dev_grad(a)) return 0;
            if (!tc__cuda_accumulate(a->dev_grad, tensor->dev_grad, a->size))
                return 0;
            nd_cuda_produced_grad(a);
            return 1;
        }
        default:
            return 0;
    }
}
#endif

static void backward_node(tc_tensor_nd *tensor) {
    if (!tensor || !tensor->requires_grad || !tensor->grad) return;
    tc_tensor_nd *a = tensor->parent_count > 0 ? tensor->parents[0] : NULL;
    tc_tensor_nd *b = tensor->parent_count > 1 ? tensor->parents[1] : NULL;
#ifdef TENSORC_HAS_CUDA
    if (nd_cuda_backward_node(tensor, a, b)) return;
    /* Camino genérico: todas las copias que el switch lee o escribe deben
     * estar al día en host (data y grad del nodo y de cada padre). */
    nd_cuda_host_grad(tensor);
    nd_cuda_host_data(tensor);
    for (size_t p = 0; p < tensor->parent_count; ++p) {
        nd_cuda_host_data(tensor->parents[p]);
        nd_cuda_host_grad(tensor->parents[p]);
    }
#endif
    switch (tensor->op) {
        case TC_ND_OP_ADD: {
            size_t cols = 0;
            /* Sesgo difundido por filas: a es [rows,cols], b es [1,cols]. */
            if (is_row_broadcast(a, b, tensor, &cols)) {
                const size_t rows = tensor->shape[0];
                if (a->requires_grad && a->grad)
                    for (size_t i = 0; i < tensor->size; ++i)
                        a->grad[i] += tensor->grad[i];
                if (b->requires_grad && b->grad)
                    for (size_t i = 0; i < rows; ++i)
                        for (size_t j = 0; j < cols; ++j)
                            b->grad[j] += tensor->grad[i * cols + j];
                break;
            }
            if (is_row_broadcast(b, a, tensor, &cols)) {
                const size_t rows = tensor->shape[0];
                if (b->requires_grad && b->grad)
                    for (size_t i = 0; i < tensor->size; ++i)
                        b->grad[i] += tensor->grad[i];
                if (a->requires_grad && a->grad)
                    for (size_t i = 0; i < rows; ++i)
                        for (size_t j = 0; j < cols; ++j)
                            a->grad[j] += tensor->grad[i * cols + j];
                break;
            }
            for (size_t i = 0; i < tensor->size; ++i) {
                add_grad(a, broadcast_index(a, tensor, i), tensor->grad[i]);
                add_grad(b, broadcast_index(b, tensor, i), tensor->grad[i]);
            }
            break;
        }
        case TC_ND_OP_MUL: {
            size_t cols = 0;
            if (is_row_broadcast(a, b, tensor, &cols)) {
                const size_t rows = tensor->shape[0];
                if (a->requires_grad && a->grad)
                    for (size_t i = 0; i < rows; ++i)
                        for (size_t j = 0; j < cols; ++j)
                            a->grad[i * cols + j] +=
                                tensor->grad[i * cols + j] * b->data[j];
                if (b->requires_grad && b->grad)
                    for (size_t i = 0; i < rows; ++i)
                        for (size_t j = 0; j < cols; ++j)
                            b->grad[j] +=
                                tensor->grad[i * cols + j] * a->data[i * cols + j];
                break;
            }
            if (is_row_broadcast(b, a, tensor, &cols)) {
                const size_t rows = tensor->shape[0];
                if (b->requires_grad && b->grad)
                    for (size_t i = 0; i < rows; ++i)
                        for (size_t j = 0; j < cols; ++j)
                            b->grad[i * cols + j] +=
                                tensor->grad[i * cols + j] * a->data[j];
                if (a->requires_grad && a->grad)
                    for (size_t i = 0; i < rows; ++i)
                        for (size_t j = 0; j < cols; ++j)
                            a->grad[j] +=
                                tensor->grad[i * cols + j] * b->data[i * cols + j];
                break;
            }
            for (size_t i = 0; i < tensor->size; ++i) {
                const size_t a_index = broadcast_index(a, tensor, i);
                const size_t b_index = broadcast_index(b, tensor, i);
                add_grad(a, a_index, tensor->grad[i] * b->data[b_index]);
                add_grad(b, b_index, tensor->grad[i] * a->data[a_index]);
            }
            break;
        }
        case TC_ND_OP_MATMUL: {
#ifdef TENSORC_HAS_ACCELERATE
            /* dA = grad @ Bᵀ y dB = Aᵀ @ grad vía GEMM; el resultado se
             * acumula sobre los gradientes existentes, igual que la ruta
             * genérica. Si la memoria temporal falla se cae al loop. */
            const size_t m = a->shape[0], k = a->shape[1], n = b->shape[1];
            if (m <= (size_t)INT_MAX && k <= (size_t)INT_MAX &&
                n <= (size_t)INT_MAX && m > 0 && k > 0 && n > 0 &&
                m <= ((size_t)-1) / n && k <= ((size_t)-1) / n &&
                m * n * k >= ((size_t)1 << 12)) {
                const int want_a = a->requires_grad && a->grad;
                const int want_b = b->requires_grad && b->grad;
                float *tmp_a = want_a
                    ? (float *)malloc(m * k * sizeof(float)) : NULL;
                float *tmp_b = want_b
                    ? (float *)malloc(k * n * sizeof(float)) : NULL;
                if ((want_a && !tmp_a) || (want_b && !tmp_b)) {
                    free(tmp_a);
                    free(tmp_b);
                } else {
                    if (tmp_a) {
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                                    (int)m, (int)k, (int)n,
                                    1.0f, tensor->grad, (int)n,
                                    b->data, (int)n,
                                    0.0f, tmp_a, (int)k);
                        for (size_t i = 0; i < m * k; ++i)
                            a->grad[i] += tmp_a[i];
                    }
                    if (tmp_b) {
                        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                                    (int)k, (int)n, (int)m,
                                    1.0f, a->data, (int)k,
                                    tensor->grad, (int)n,
                                    0.0f, tmp_b, (int)n);
                        for (size_t i = 0; i < k * n; ++i)
                            b->grad[i] += tmp_b[i];
                    }
                    free(tmp_a);
                    free(tmp_b);
                    break;
                }
            }
#endif
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
        }
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
        case TC_ND_OP_AVGPOOL2D: {
            const size_t pool_h = a->shape[2] / tensor->shape[2];
            const size_t pool_w = a->shape[3] / tensor->shape[3];
            const size_t batch = a->shape[0];
            const size_t channels = a->shape[1];
            const size_t input_height = a->shape[2];
            const size_t input_width = a->shape[3];
            const size_t output_height = tensor->shape[2];
            const size_t output_width = tensor->shape[3];
            const float share = 1.0f / (float)(pool_h * pool_w);
            for (size_t n = 0; n < batch; ++n)
                for (size_t c = 0; c < channels; ++c)
                    for (size_t oh = 0; oh < output_height; ++oh)
                        for (size_t ow = 0; ow < output_width; ++ow) {
                            const size_t row0 = oh * pool_h;
                            const size_t col0 = ow * pool_w;
                            const float upstream =
                                tensor->grad[(((n * channels + c) * output_height + oh) *
                                              output_width + ow)] * share;
                            for (size_t ph = 0; ph < pool_h; ++ph)
                                for (size_t pw = 0; pw < pool_w; ++pw)
                                    add_grad(a,
                                             (((n * channels + c) * input_height + row0 + ph) *
                                              input_width + col0 + pw),
                                             upstream);
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
        case TC_ND_OP_PERMUTE: {
            /* El forward es una biyección: cada índice lineal de la
             * entrada contribuye su gradiente desde la posición permutada
             * de la salida (mismo mapeo que en el forward). */
            size_t in_idx[8];
            for (size_t lin = 0; lin < a->size; ++lin) {
                for (size_t d = 0; d < a->ndim; ++d)
                    in_idx[d] = (lin / a->strides[d]) % a->shape[d];
                size_t out_lin = 0;
                for (size_t k = 0; k < a->ndim; ++k)
                    out_lin += in_idx[tensor->perm[k]] * tensor->strides[k];
                add_grad(a, lin, tensor->grad[out_lin]);
            }
            break;
        }
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
#ifdef TENSORC_HAS_CUDA
    /* El backward genérico acumuló gradientes en host (la atención escribe
     * q/k/v->grad directamente, sin pasar por add_grad): las copias de
     * device de los padres quedan obsoletas. */
    for (size_t p = 0; p < tensor->parent_count; ++p)
        tensor->parents[p]->dev_grad_valid = 0;
#endif
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
#ifdef TENSORC_HAS_CUDA
    /* La semilla se escribe en host: esa copia manda sobre la de device
     * (zero_grad pudo haber dejado la copia de device como la válida). */
    loss->host_grad_valid = 1;
    loss->dev_grad_valid = 0;
#endif
    for (size_t i = topology.count; i > 0; --i) backward_node(topology.items[i - 1]);
    free(topology.items);
    return 1;
}

void tc_tensor_nd_print(const tc_tensor_nd *tensor, const char *name) {
    if (!tensor || !tensor->data) {
        printf("%s: <tensor inválido>\n", name ? name : "tensor");
        return;
    }
    ND_SYNC_DATA(tensor);
    printf("%s shape=[", name ? name : "tensor");
    for (size_t i = 0; i < tensor->ndim; ++i)
        printf("%s%zu", i ? ", " : "", tensor->shape[i]);
    printf("] values=");
    for (size_t i = 0; i < tensor->size; ++i) printf("%s%.5f", i ? ", " : "[", tensor->data[i]);
    printf("]\n");
}

void tc_tensor_nd_sgd_step(tc_tensor_nd *param, float learning_rate) {
    if (!param || !param->grad) return;
#ifdef TENSORC_HAS_CUDA
    if (tc_cuda_available() && nd_cuda_dev_data(param) &&
        nd_cuda_dev_grad(param) &&
        tc__cuda_sgd_step(param->dev_data, param->dev_grad, param->size,
                          learning_rate)) {
        nd_cuda_produced_data(param);
        return;
    }
    nd_cuda_host_data(param);
    nd_cuda_host_grad(param);
#endif
    for (size_t i = 0; i < param->size; ++i)
        param->data[i] -= learning_rate * param->grad[i];
#ifdef TENSORC_HAS_CUDA
    param->dev_data_valid = 0;
#endif
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
#ifdef TENSORC_HAS_CUDA
    /* Momentos residentes en device; mv_on_device[i]=1 significa que la
     * copia al día de m[i]/v[i] es la de device (la de host está obsoleta
     * y se resincroniza si un paso debe caer a CPU). */
    float **dev_m;
    float **dev_v;
    unsigned char *mv_on_device;
#endif
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
#ifdef TENSORC_HAS_CUDA
    optimizer->dev_m = (float **)calloc(count, sizeof(float *));
    optimizer->dev_v = (float **)calloc(count, sizeof(float *));
    optimizer->mv_on_device =
        (unsigned char *)calloc(count ? count : 1, sizeof(unsigned char));
    if ((count > 0) &&
        (!optimizer->dev_m || !optimizer->dev_v || !optimizer->mv_on_device)) {
        tc_tensor_nd_optimizer_free(optimizer);
        return NULL;
    }
#endif
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
#ifdef TENSORC_HAS_CUDA
        /* Ruta GPU: los momentos viven en device junto al parámetro; el
         * host solo se resincroniza si hay que caer al camino de CPU. */
        if (tc_cuda_available()) {
            if (!optimizer->dev_m[i]) {
                optimizer->dev_m[i] = tc__cuda_alloc(param->size);
                if (optimizer->dev_m[i] &&
                    !tc__cuda_copy_h2d(optimizer->dev_m[i], m, param->size)) {
                    tc__cuda_free(optimizer->dev_m[i], param->size);
                    optimizer->dev_m[i] = NULL;
                }
            }
            if (!optimizer->dev_v[i]) {
                optimizer->dev_v[i] = tc__cuda_alloc(param->size);
                if (optimizer->dev_v[i] &&
                    !tc__cuda_copy_h2d(optimizer->dev_v[i], v, param->size)) {
                    tc__cuda_free(optimizer->dev_v[i], param->size);
                    optimizer->dev_v[i] = NULL;
                }
            }
            if (optimizer->dev_m[i] && optimizer->dev_v[i] &&
                nd_cuda_dev_data(param) && nd_cuda_dev_grad(param)) {
                if (!tc__cuda_adam_step(
                        param->dev_data, param->dev_grad,
                        optimizer->dev_m[i], optimizer->dev_v[i],
                        param->size, optimizer->learning_rate, beta1, beta2,
                        bias1, bias2, epsilon)) {
                    fprintf(stderr,
                            "TensorC CUDA: adam_step falló a mitad del "
                            "paso; parámetros irrecuperables.\n");
                    abort();
                }
                optimizer->mv_on_device[i] = 1;
                nd_cuda_produced_data(param);
                ++updated;
                continue;
            }
        }
        if (optimizer->mv_on_device[i]) {
            /* Caída a CPU con momentos en device: bajarlos primero. */
            if (optimizer->dev_m[i] &&
                !tc__cuda_copy_d2h(m, optimizer->dev_m[i], param->size))
                abort();
            if (optimizer->dev_v[i] &&
                !tc__cuda_copy_d2h(v, optimizer->dev_v[i], param->size))
                abort();
            optimizer->mv_on_device[i] = 0;
        }
        nd_cuda_host_data(param);
        nd_cuda_host_grad(param);
#endif
        for (size_t j = 0; j < param->size; ++j) {
            const float grad = param->grad[j];
            m[j] = beta1 * m[j] + (1.0f - beta1) * grad;
            v[j] = beta2 * v[j] + (1.0f - beta2) * grad * grad;
            const float m_hat = m[j] / bias1;
            const float v_hat = v[j] / bias2;
            param->data[j] -=
                optimizer->learning_rate * m_hat / (sqrtf(v_hat) + epsilon);
        }
#ifdef TENSORC_HAS_CUDA
        param->dev_data_valid = 0;
#endif
        ++updated;
    }
    return updated;
}

void tc_tensor_nd_optimizer_free(tc_tensor_nd_optimizer *optimizer) {
    if (!optimizer) return;
    for (size_t i = 0; i < optimizer->count; ++i) {
        free(optimizer->m ? optimizer->m[i] : NULL);
        free(optimizer->v ? optimizer->v[i] : NULL);
#ifdef TENSORC_HAS_CUDA
        const size_t psize =
            optimizer->params[i] ? optimizer->params[i]->size : 0;
        tc__cuda_free(optimizer->dev_m ? optimizer->dev_m[i] : NULL, psize);
        tc__cuda_free(optimizer->dev_v ? optimizer->dev_v[i] : NULL, psize);
#endif
    }
#ifdef TENSORC_HAS_CUDA
    free(optimizer->dev_m);
    free(optimizer->dev_v);
    free(optimizer->mv_on_device);
#endif
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

/* --- Atención multi-cabeza y bloque Transformer --- */

/* [batch, seq, depth] -> [batch*heads, seq, depth/heads]. */
static tc_tensor_nd *split_heads(const tc_tensor_nd *t, size_t heads) {
    const size_t batch = t->shape[0];
    const size_t seq = t->shape[1];
    const size_t depth = t->shape[2];
    const size_t head_depth = depth / heads;
    const size_t s1[] = {batch, seq, heads, head_depth};
    tc_tensor_nd *a = tc_tensor_nd_reshape_to(t, s1, 4);
    if (!a) return NULL;
    const size_t axes[] = {0, 2, 1, 3};
    tc_tensor_nd *b = tc_tensor_nd_permute(a, axes, 4);
    tc_tensor_nd_free(a);
    if (!b) return NULL;
    const size_t s2[] = {batch * heads, seq, head_depth};
    tc_tensor_nd *c = tc_tensor_nd_reshape_to(b, s2, 3);
    tc_tensor_nd_free(b);
    return c;
}

/* [batch*heads, seq, depth/heads] -> [batch, seq, depth]. */
static tc_tensor_nd *merge_heads(const tc_tensor_nd *t, size_t batch,
                                 size_t seq, size_t depth, size_t heads) {
    const size_t head_depth = depth / heads;
    const size_t s1[] = {batch, heads, seq, head_depth};
    tc_tensor_nd *a = tc_tensor_nd_reshape_to(t, s1, 4);
    if (!a) return NULL;
    const size_t axes[] = {0, 2, 1, 3};
    tc_tensor_nd *b = tc_tensor_nd_permute(a, axes, 4);
    tc_tensor_nd_free(a);
    if (!b) return NULL;
    const size_t s2[] = {batch, seq, depth};
    tc_tensor_nd *c = tc_tensor_nd_reshape_to(b, s2, 3);
    tc_tensor_nd_free(b);
    return c;
}

tc_tensor_nd *tc_tensor_nd_multihead_attention(const tc_tensor_nd *query,
                                               const tc_tensor_nd *key,
                                               const tc_tensor_nd *value,
                                               size_t heads, int causal) {
    if (!query || !key || !value || heads == 0 ||
        query->ndim != 3 || key->ndim != 3 || value->ndim != 3 ||
        query->shape[0] != key->shape[0] ||
        query->shape[0] != value->shape[0] ||
        query->shape[1] != key->shape[1] ||
        key->shape[1] != value->shape[1] ||
        query->shape[2] != key->shape[2] ||
        query->shape[2] % heads != 0 ||
        value->shape[2] % heads != 0)
        return NULL;
    const size_t batch = query->shape[0];
    const size_t seq = query->shape[1];
    const size_t depth = query->shape[2];

    tc_tensor_nd *qh = split_heads(query, heads);
    tc_tensor_nd *kh = split_heads(key, heads);
    tc_tensor_nd *vh = split_heads(value, heads);
    if (!qh || !kh || !vh) {
        tc_tensor_nd_free(qh);
        tc_tensor_nd_free(kh);
        tc_tensor_nd_free(vh);
        return NULL;
    }
    tc_tensor_nd *attn = tc_tensor_nd_attention(qh, kh, vh, causal);
    tc_tensor_nd_free(qh);
    tc_tensor_nd_free(kh);
    tc_tensor_nd_free(vh);
    if (!attn) return NULL;
    tc_tensor_nd *merged = merge_heads(attn, batch, seq, depth, heads);
    tc_tensor_nd_free(attn);
    return merged;
}

tc_tensor_nd *tc_tensor_nd_sinusoidal_positions(size_t seq, size_t depth) {
    if (seq == 0 || depth == 0) return NULL;
    const size_t shape[] = {seq, depth};
    tc_tensor_nd *pe = tc_tensor_nd_create(shape, 2, 0);
    if (!pe) return NULL;
    float *data = tc_tensor_nd_data(pe);
    for (size_t p = 0; p < seq; ++p)
        for (size_t i = 0; i < depth; ++i) {
            const double angle =
                (double)p / pow(10000.0, (double)(2 * (i / 2)) / (double)depth);
            data[p * depth + i] =
                (float)(i % 2 == 0 ? sin(angle) : cos(angle));
        }
    return pe;
}

struct tc_transformer_block {
    size_t model_dim;
    size_t ff_dim;
    size_t heads;
    tc_tensor_nd *w_q;
    tc_tensor_nd *w_k;
    tc_tensor_nd *w_v;
    tc_tensor_nd *w_o;
    tc_tensor_nd *w1;
    tc_tensor_nd *b1;
    tc_tensor_nd *w2;
    tc_tensor_nd *b2;
};

/* Inicializa `weight` con Xavier uniforme determinista a partir de `state`. */
static void xavier_fill(tc_tensor_nd *weight, size_t fan_in, size_t fan_out,
                        unsigned int *state) {
    const float limit =
        sqrtf(6.0f / ((float)fan_in + (float)fan_out));
    float *data = tc_tensor_nd_data(weight);
    const size_t count = tc_tensor_nd_numel(weight);
    for (size_t i = 0; i < count; ++i) {
        const float unit = (float)layer_rng_next(state) / 4294967296.0f;
        data[i] = (2.0f * unit - 1.0f) * limit;
    }
}

static tc_tensor_nd *transformer_new_weight(size_t rows, size_t cols,
                                            unsigned int *state) {
    const size_t shape[] = {rows, cols};
    tc_tensor_nd *w = tc_tensor_nd_create(shape, 2, 1);
    if (w) xavier_fill(w, rows, cols, state);
    return w;
}

static tc_tensor_nd *transformer_new_bias(size_t cols) {
    const size_t shape[] = {1, cols};
    return tc_tensor_nd_create(shape, 2, 1); /* datos ya en cero */
}

tc_transformer_block *tc_transformer_block_create(size_t model_dim,
                                                  size_t ff_dim,
                                                  size_t heads,
                                                  unsigned int seed) {
    if (model_dim == 0 || ff_dim == 0 || heads == 0 ||
        model_dim % heads != 0)
        return NULL;
    tc_transformer_block *block =
        (tc_transformer_block *)calloc(1, sizeof(*block));
    if (!block) return NULL;
    block->model_dim = model_dim;
    block->ff_dim = ff_dim;
    block->heads = heads;
    unsigned int state = seed ? seed : 1u;
    block->w_q = transformer_new_weight(model_dim, model_dim, &state);
    block->w_k = transformer_new_weight(model_dim, model_dim, &state);
    block->w_v = transformer_new_weight(model_dim, model_dim, &state);
    block->w_o = transformer_new_weight(model_dim, model_dim, &state);
    block->w1 = transformer_new_weight(model_dim, ff_dim, &state);
    block->b1 = transformer_new_bias(ff_dim);
    block->w2 = transformer_new_weight(ff_dim, model_dim, &state);
    block->b2 = transformer_new_bias(model_dim);
    if (!block->w_q || !block->w_k || !block->w_v || !block->w_o ||
        !block->w1 || !block->b1 || !block->w2 || !block->b2) {
        tc_transformer_block_free(block);
        return NULL;
    }
    return block;
}

void tc_transformer_block_free(tc_transformer_block *block) {
    if (!block) return;
    tc_tensor_nd_free(block->w_q);
    tc_tensor_nd_free(block->w_k);
    tc_tensor_nd_free(block->w_v);
    tc_tensor_nd_free(block->w_o);
    tc_tensor_nd_free(block->w1);
    tc_tensor_nd_free(block->b1);
    tc_tensor_nd_free(block->w2);
    tc_tensor_nd_free(block->b2);
    free(block);
}

size_t tc_transformer_block_param_count(const tc_transformer_block *block) {
    (void)block;
    return 8;
}

tc_tensor_nd *tc_transformer_block_param(tc_transformer_block *block,
                                         size_t index) {
    if (!block || index >= 8) return NULL;
    tc_tensor_nd *params[] = {block->w_q, block->w_k, block->w_v, block->w_o,
                              block->w1,  block->b1,  block->w2, block->b2};
    return params[index];
}

/*
 * Pre-norm: x + MHA(LN(x)) y x + FFN(LN(x)). Las proyecciones y la FFN se
 * calculan en 2D [batch*seq, model_dim]; solo la atención vuelve a 3D.
 * Todas las operaciones son nodos de grafo: el backward es el estándar.
 */
tc_tensor_nd *tc_transformer_block_forward(tc_transformer_block *block,
                                           const tc_tensor_nd *input) {
    if (!block || !input || input->ndim != 3 ||
        input->shape[2] != block->model_dim)
        return NULL;
    const size_t batch = input->shape[0];
    const size_t seq = input->shape[1];
    const size_t dim = block->model_dim;
    const size_t flat = batch * seq;
    const size_t x2_shape[] = {flat, dim};
    const size_t x3_shape[] = {batch, seq, dim};

    tc_tensor_nd *x2 = tc_tensor_nd_reshape_to(input, x2_shape, 2);
    if (!x2) return NULL;
    tc_tensor_nd *ln1 = tc_tensor_nd_layer_norm(x2);
    tc_tensor_nd *q2 = ln1 ? tc_tensor_nd_matmul(ln1, block->w_q) : NULL;
    tc_tensor_nd *k2 = ln1 ? tc_tensor_nd_matmul(ln1, block->w_k) : NULL;
    tc_tensor_nd *v2 = ln1 ? tc_tensor_nd_matmul(ln1, block->w_v) : NULL;
    tc_tensor_nd_free(ln1);
    tc_tensor_nd *q = q2 ? tc_tensor_nd_reshape_to(q2, x3_shape, 3) : NULL;
    tc_tensor_nd *k = k2 ? tc_tensor_nd_reshape_to(k2, x3_shape, 3) : NULL;
    tc_tensor_nd *v = v2 ? tc_tensor_nd_reshape_to(v2, x3_shape, 3) : NULL;
    tc_tensor_nd_free(q2);
    tc_tensor_nd_free(k2);
    tc_tensor_nd_free(v2);
    tc_tensor_nd *attn =
        (q && k && v)
            ? tc_tensor_nd_multihead_attention(q, k, v, block->heads, 0)
            : NULL;
    tc_tensor_nd_free(q);
    tc_tensor_nd_free(k);
    tc_tensor_nd_free(v);
    tc_tensor_nd *attn2 =
        attn ? tc_tensor_nd_reshape_to(attn, x2_shape, 2) : NULL;
    tc_tensor_nd_free(attn);
    tc_tensor_nd *proj =
        attn2 ? tc_tensor_nd_matmul(attn2, block->w_o) : NULL;
    tc_tensor_nd_free(attn2);
    tc_tensor_nd *h1 = (x2 && proj) ? tc_tensor_nd_add(x2, proj) : NULL;
    tc_tensor_nd_free(proj);

    tc_tensor_nd *ln2 = h1 ? tc_tensor_nd_layer_norm(h1) : NULL;
    tc_tensor_nd *ff1a = ln2 ? tc_tensor_nd_matmul(ln2, block->w1) : NULL;
    tc_tensor_nd_free(ln2);
    tc_tensor_nd *ff1 = (ff1a && block->b1)
                            ? tc_tensor_nd_add(ff1a, block->b1) : NULL;
    tc_tensor_nd_free(ff1a);
    tc_tensor_nd *ff2 = ff1 ? tc_tensor_nd_relu(ff1) : NULL;
    tc_tensor_nd_free(ff1);
    tc_tensor_nd *ff3a = ff2 ? tc_tensor_nd_matmul(ff2, block->w2) : NULL;
    tc_tensor_nd_free(ff2);
    tc_tensor_nd *ff3 = (ff3a && block->b2)
                            ? tc_tensor_nd_add(ff3a, block->b2) : NULL;
    tc_tensor_nd_free(ff3a);
    tc_tensor_nd *out2 = (h1 && ff3) ? tc_tensor_nd_add(h1, ff3) : NULL;
    tc_tensor_nd_free(ff3);
    tc_tensor_nd *out =
        out2 ? tc_tensor_nd_reshape_to(out2, x3_shape, 3) : NULL;
    tc_tensor_nd_free(out2);
    tc_tensor_nd_free(h1);
    tc_tensor_nd_free(x2);
    return out;
}

/* ------------------------------------------------------------------ */
/* Serialización binaria de tensores N-D.                              */
/*                                                                     */
/* Formatos (enteros de 64 bits y float32 en orden de bytes del host): */
/*   tensor único:  magic "TCND0001" | ndim | shape[ndim] | datos      */
/*   checkpoint:    magic "TCKP0001" | count | tensor × count          */
/* Los gradientes no se guardan; al cargar se elige requires_grad.     */
/* ------------------------------------------------------------------ */

#define TC_ND_SAVE_MAGIC_TENSOR "TCND0001"
#define TC_ND_SAVE_MAGIC_CHECKPOINT "TCKP0001"

static int write_u64(FILE *file, uint64_t value) {
    return fwrite(&value, sizeof(value), 1, file) == 1;
}

static int read_u64(FILE *file, uint64_t *value) {
    return fread(value, sizeof(*value), 1, file) == 1;
}

static int write_tensor_payload(FILE *file, const tc_tensor_nd *tensor) {
    ND_SYNC_DATA(tensor);
    if (!write_u64(file, (uint64_t)tensor->ndim)) return 0;
    for (size_t i = 0; i < tensor->ndim; ++i)
        if (!write_u64(file, (uint64_t)tensor->shape[i])) return 0;
    return fwrite(tensor->data, sizeof(float), tensor->size, file) ==
           tensor->size;
}

/* Lee la forma y los datos de un tensor; devuelve el tensor listo o
 * NULL si el archivo está truncado, es corrupto o no hay memoria. */
static tc_tensor_nd *read_tensor_payload(FILE *file, int requires_grad) {
    uint64_t ndim = 0;
    if (!read_u64(file, &ndim) || ndim == 0 || ndim > 8) return NULL;
    size_t shape[8];
    for (size_t i = 0; i < (size_t)ndim; ++i) {
        uint64_t dim = 0;
        if (!read_u64(file, &dim) || dim == 0 ||
            dim > (uint64_t)((size_t)-1)) {
            return NULL;
        }
        shape[i] = (size_t)dim;
    }
    size_t size = 0;
    if (!checked_product(shape, (size_t)ndim, &size)) return NULL;
    float *data = (float *)malloc(size * sizeof(float));
    if (!data) return NULL;
    if (fread(data, sizeof(float), size, file) != size) {
        free(data);
        return NULL;
    }
    tc_tensor_nd *tensor =
        tc_tensor_nd_from_data(shape, (size_t)ndim, data, requires_grad);
    free(data);
    return tensor;
}

int tc_tensor_nd_save(const tc_tensor_nd *tensor, const char *path) {
    if (!tensor || !path) return 0;
    FILE *file = fopen(path, "wb");
    if (!file) return 0;
    int ok = fwrite(TC_ND_SAVE_MAGIC_TENSOR, 1, 8, file) == 8 &&
             write_tensor_payload(file, tensor);
    if (fclose(file) != 0) ok = 0;
    return ok;
}

tc_tensor_nd *tc_tensor_nd_load(const char *path, int requires_grad) {
    if (!path) return NULL;
    FILE *file = fopen(path, "rb");
    if (!file) return NULL;
    char magic[8];
    tc_tensor_nd *tensor = NULL;
    if (fread(magic, 1, 8, file) == 8 &&
        memcmp(magic, TC_ND_SAVE_MAGIC_TENSOR, 8) == 0)
        tensor = read_tensor_payload(file, requires_grad);
    fclose(file);
    return tensor;
}

int tc_save_checkpoint(const tc_tensor_nd *const *tensors, size_t count,
                       const char *path) {
    if ((!tensors && count > 0) || !path) return 0;
    FILE *file = fopen(path, "wb");
    if (!file) return 0;
    int ok = fwrite(TC_ND_SAVE_MAGIC_CHECKPOINT, 1, 8, file) == 8 &&
             write_u64(file, (uint64_t)count);
    for (size_t i = 0; ok && i < count; ++i)
        ok = tensors[i] && write_tensor_payload(file, tensors[i]);
    if (fclose(file) != 0) ok = 0;
    return ok;
}

tc_tensor_nd **tc_load_checkpoint(const char *path, size_t *count,
                                  int requires_grad) {
    if (count) *count = 0;
    if (!path) return NULL;
    FILE *file = fopen(path, "rb");
    if (!file) return NULL;
    char magic[8];
    tc_tensor_nd **tensors = NULL;
    uint64_t saved_count = 0;
    if (fread(magic, 1, 8, file) != 8 ||
        memcmp(magic, TC_ND_SAVE_MAGIC_CHECKPOINT, 8) != 0 ||
        !read_u64(file, &saved_count) || saved_count > (uint64_t)SIZE_MAX) {
        fclose(file);
        return NULL;
    }
    tensors = (tc_tensor_nd **)calloc((size_t)saved_count + 1,
                                      sizeof(*tensors));
    if (!tensors) {
        fclose(file);
        return NULL;
    }
    size_t loaded = 0;
    for (; loaded < (size_t)saved_count; ++loaded) {
        tensors[loaded] = read_tensor_payload(file, requires_grad);
        if (!tensors[loaded]) {
            tc_free_checkpoint(tensors, loaded);
            fclose(file);
            return NULL;
        }
    }
    fclose(file);
    if (count) *count = loaded;
    return tensors;
}

void tc_free_checkpoint(tc_tensor_nd **tensors, size_t count) {
    if (!tensors) return;
    for (size_t i = 0; i < count; ++i)
        tc_tensor_nd_free(tensors[i]);
    free(tensors);
}

#ifndef TENSORC_HAS_CUDA
/* Stubs de la API pública CUDA cuando la biblioteca se compila sin el
 * backend: la API existe siempre y reporta "no disponible". Con
 * TENSORC_HAS_CUDA las implementaciones viven en tensorc_cuda.cu. */
int tc_cuda_available(void) { return 0; }
int tc_cuda_set_device(int device) { (void)device; return 0; }
int tc_cuda_device(void) { return -1; }
#endif
