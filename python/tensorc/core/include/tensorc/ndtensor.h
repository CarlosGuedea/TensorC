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
/* Copia los datos de src en dst (misma forma exacta); no toca gradientes. */
int tc_tensor_nd_copy_(tc_tensor_nd *dst, const tc_tensor_nd *src);

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
/* Elementwise sigmoid: 1 / (1 + exp(-x)), numerically stable. */
tc_tensor_nd *tc_tensor_nd_sigmoid(const tc_tensor_nd *input);
/* Elementwise hyperbolic tangent. */
tc_tensor_nd *tc_tensor_nd_tanh(const tc_tensor_nd *input);
/* Softmax over the last dimension (one distribution per row). */
tc_tensor_nd *tc_tensor_nd_softmax(const tc_tensor_nd *input);
/*
 * Mean cross-entropy over rows: prediction holds softmax probabilities and
 * target holds one-hot rows; both tensors must have the same shape.
 */
tc_tensor_nd *tc_tensor_nd_cross_entropy(const tc_tensor_nd *prediction,
                                         const tc_tensor_nd *target);
/*
 * Embedding: gather de filas de `table` [vocab, dim] según los índices
 * enteros de `ids` (floats exactos); la salida es ids->shape + [dim].
 * Solo la tabla recibe gradiente (scatter-add por fila).
 */
tc_tensor_nd *tc_tensor_nd_embedding(const tc_tensor_nd *table,
                                     const tc_tensor_nd *ids);
/*
 * Cross-entropy con objetivos enteros: prediction [rows, classes] trae
 * probabilidades e ids [rows] la clase correcta de cada fila. Equivale a
 * cross_entropy con one-hot sin materializar los objetivos; ids no recibe
 * gradiente.
 */
tc_tensor_nd *tc_tensor_nd_cross_entropy_sparse(const tc_tensor_nd *prediction,
                                                const tc_tensor_nd *ids);
/* Reshapes as a graph operation; the total element count must match. */
tc_tensor_nd *tc_tensor_nd_reshape_to(const tc_tensor_nd *input,
                                      const size_t *shape, size_t ndim);
/*
 * Non-overlapping max pooling over the spatial axes of a [N,C,H,W] tensor.
 * Both H and W must be divisible by the pool size; the output is
 * [N, C, H/pool_h, W/pool_w].
 */
tc_tensor_nd *tc_tensor_nd_maxpool2d(const tc_tensor_nd *input,
                                     size_t pool_h, size_t pool_w);
/*
 * Non-overlapping average pooling over the spatial axes of a [N,C,H,W]
 * tensor. Same shape contract as maxpool2d: the output is
 * [N, C, H/pool_h, W/pool_w] and the backward divides the upstream gradient
 * evenly among the window's elements.
 */
tc_tensor_nd *tc_tensor_nd_avgpool2d(const tc_tensor_nd *input,
                                     size_t pool_h, size_t pool_w);
tc_tensor_nd *tc_tensor_nd_sum(const tc_tensor_nd *input);
tc_tensor_nd *tc_tensor_nd_mse_loss(const tc_tensor_nd *prediction,
                                    const tc_tensor_nd *target);

/* Slice contiguo a lo largo de un eje (start + length <= shape[axis]). */
tc_tensor_nd *tc_tensor_nd_slice_axis(const tc_tensor_nd *input, size_t axis,
                                      size_t start, size_t length);
/* Concatena N tensores en un eje; todas las demás dimensiones deben coincidir. */
tc_tensor_nd *tc_tensor_nd_concat(const tc_tensor_nd *const *inputs,
                                  size_t count, size_t axis);
/* LayerNorm sobre la última dimensión (sin afín; componer con mul/add). */
tc_tensor_nd *tc_tensor_nd_layer_norm(const tc_tensor_nd *input);
/* Transposición de matriz [M, N] -> [N, M]. */
tc_tensor_nd *tc_tensor_nd_transpose(const tc_tensor_nd *input);
/*
 * Atención single-head: softmax((Q Kᵀ)/sqrt(depth) + máscara) V sobre
 * tensores [batch, seq, depth]. causal=1 enmascara posiciones futuras.
 */
tc_tensor_nd *tc_tensor_nd_attention(const tc_tensor_nd *query,
                                     const tc_tensor_nd *key,
                                     const tc_tensor_nd *value,
                                     int causal);
/* Permutación general de ejes: out[i] = in[axes[i]] por dimensión. */
tc_tensor_nd *tc_tensor_nd_permute(const tc_tensor_nd *input,
                                   const size_t *axes, size_t ndim);
/*
 * Atención multi-cabeza sobre [batch, seq, depth] con `heads` cabezas
 * (depth debe ser divisible entre heads). Compone permute + reshape +
 * atención single-head + concat, así el autograd fluye sin nodos nuevos.
 */
tc_tensor_nd *tc_tensor_nd_multihead_attention(const tc_tensor_nd *query,
                                               const tc_tensor_nd *key,
                                               const tc_tensor_nd *value,
                                               size_t heads, int causal);
/* Codificación posicional sinusoidal [seq, depth] (hoja constante). */
tc_tensor_nd *tc_tensor_nd_sinusoidal_positions(size_t seq, size_t depth);

/*
 * Serialización binaria de tensores N-D (formato propio, versionado,
 * documentado en ndtensor.c). Guarda ndim, shape y datos — nunca
 * gradientes —, así sirve para persistir pesos entrenados. Los enteros
 * son de 64 bits y los datos float32, en orden de bytes del host.
 * Devuelven 0/NULL si el archivo no existe o está corrupto.
 */
int tc_tensor_nd_save(const tc_tensor_nd *tensor, const char *path);
tc_tensor_nd *tc_tensor_nd_load(const char *path, int requires_grad);
/*
 * Checkpoint: varios tensores en un solo archivo (la lista de parámetros
 * de un modelo, p. ej.). tc_load_checkpoint devuelve un arreglo de
 * `count` tensores que hay que liberar con tc_free_checkpoint.
 */
int tc_save_checkpoint(const tc_tensor_nd *const *tensors, size_t count,
                       const char *path);
tc_tensor_nd **tc_load_checkpoint(const char *path, size_t *count,
                                  int requires_grad);
void tc_free_checkpoint(tc_tensor_nd **tensors, size_t count);

/* Runs reverse-mode autodiff from a scalar loss. Returns 1 on success. */
int tc_tensor_nd_backward(tc_tensor_nd *loss);
void tc_tensor_nd_print(const tc_tensor_nd *tensor, const char *name);

/* Applies one SGD update: data -= learning_rate * grad. */
void tc_tensor_nd_sgd_step(tc_tensor_nd *param, float learning_rate);

/*
 * Adam over a fixed parameter list (betas 0.9 / 0.999, epsilon 1e-8).
 * The optimizer copies the pointer array; the caller keeps owning the
 * tensors. Each step updates every parameter that carries a gradient.
 */
typedef struct tc_tensor_nd_optimizer tc_tensor_nd_optimizer;
tc_tensor_nd_optimizer *tc_tensor_nd_optimizer_adam(tc_tensor_nd **params,
                                                    size_t count,
                                                    float learning_rate);
/* Applies one Adam step. Returns the number of updated parameters. */
int tc_tensor_nd_optimizer_step(tc_tensor_nd_optimizer *optimizer);
/* Cambia el learning rate en caliente (warmup, decaimiento, schedules). */
void tc_tensor_nd_optimizer_set_lr(tc_tensor_nd_optimizer *optimizer,
                                   float learning_rate);
void tc_tensor_nd_optimizer_free(tc_tensor_nd_optimizer *optimizer);

/*
 * High-level 2D convolution layer: kernel [K,C,KH,KW] plus bias [1,K,1,1],
 * both trainable. The kernel uses Xavier uniform initialization scaled by
 * sqrt(6 / (fan_in + fan_out)); the seed makes creation deterministic.
 */
typedef struct tc_conv2d_layer tc_conv2d_layer;
tc_conv2d_layer *tc_conv2d_layer_create(size_t in_channels, size_t out_channels,
                                        size_t kernel_h, size_t kernel_w,
                                        tc_conv2d_options options,
                                        unsigned int seed);
void tc_conv2d_layer_free(tc_conv2d_layer *layer);
/* Forward pass: conv2d_ex(input, kernel, options) + bias (broadcast). */
tc_tensor_nd *tc_conv2d_layer_forward(tc_conv2d_layer *layer,
                                      const tc_tensor_nd *input);
tc_tensor_nd *tc_conv2d_layer_kernel(tc_conv2d_layer *layer);
tc_tensor_nd *tc_conv2d_layer_bias(tc_conv2d_layer *layer);

/*
 * High-level fully-connected layer: weight [in_features, out_features] plus
 * bias [1, out_features], both trainable. The weight uses Xavier uniform
 * initialization scaled by sqrt(6 / (fan_in + fan_out)); the seed makes
 * creation deterministic. The input must be rank 2: [batch, in_features].
 */
typedef struct tc_dense_layer tc_dense_layer;
tc_dense_layer *tc_dense_layer_create(size_t in_features, size_t out_features,
                                      unsigned int seed);
void tc_dense_layer_free(tc_dense_layer *layer);
/* Forward pass: matmul(input, weight) + bias (broadcast over rows). */
tc_tensor_nd *tc_dense_layer_forward(tc_dense_layer *layer,
                                     const tc_tensor_nd *input);
tc_tensor_nd *tc_dense_layer_weight(tc_dense_layer *layer);
tc_tensor_nd *tc_dense_layer_bias(tc_dense_layer *layer);

/*
 * High-level LSTM layer over a full sequence. Input is [seq, batch, in]
 * (time-major); output is [seq, batch, hidden]. h0/c0 start at zero and
 * the same weights are shared across time steps (gradients accumulate).
 * Building the forward unfolds the graph, so the output retains every
 * step: release it only after backward/training use.
 */
typedef struct tc_lstm_layer tc_lstm_layer;
tc_lstm_layer *tc_lstm_layer_create(size_t input_features, size_t hidden,
                                    unsigned int seed);
void tc_lstm_layer_free(tc_lstm_layer *layer);
tc_tensor_nd *tc_lstm_layer_forward(tc_lstm_layer *layer,
                                    const tc_tensor_nd *input);
tc_tensor_nd *tc_lstm_layer_weight_ih(tc_lstm_layer *layer);
tc_tensor_nd *tc_lstm_layer_weight_hh(tc_lstm_layer *layer);
tc_tensor_nd *tc_lstm_layer_bias(tc_lstm_layer *layer);

/*
 * Bloque Transformer pre-norm: atención multi-cabeza con residuales y
 * FFN (ReLU) sobre [batch, seq, model_dim] -> misma forma. Pesos:
 * W_q/W_k/W_v/W_o [model_dim, model_dim], W1 [model_dim, ff_dim],
 * b1 [1, ff_dim], W2 [ff_dim, model_dim], b2 [1, model_dim], Xavier
 * determinista. Los gradientes se acumulan en los 8 parámetros.
 */
typedef struct tc_transformer_block tc_transformer_block;
tc_transformer_block *tc_transformer_block_create(size_t model_dim,
                                                  size_t ff_dim,
                                                  size_t heads,
                                                  unsigned int seed);
void tc_transformer_block_free(tc_transformer_block *block);
tc_tensor_nd *tc_transformer_block_forward(tc_transformer_block *block,
                                           const tc_tensor_nd *input);
size_t tc_transformer_block_param_count(const tc_transformer_block *block);
tc_tensor_nd *tc_transformer_block_param(tc_transformer_block *block,
                                         size_t index);

/*
 * Backend CUDA opcional (se compila con CMake -DTENSORC_USE_CUDA=ON).
 * La API existe en todas las compilaciones: sin el backend,
 * tc_cuda_available() devuelve 0 y tc_cuda_device() devuelve -1, así el
 * código de usuario puede consultar sin #ifdef. Las operaciones con
 * aceleración GPU se eligen automáticamente por tamaño dentro de cada op;
 * no hay que mover tensores a mano. La variable de entorno
 * TC_CUDA_MIN_ELEMS ajusta el umbral (0 fuerza GPU en todo, útil para
 * validar las rutas CUDA con la suite de pruebas).
 */
int tc_cuda_available(void);
int tc_cuda_set_device(int device); /* 1 si el dispositivo quedó activo */
int tc_cuda_device(void);           /* índice activo o -1 sin backend */

#ifdef __cplusplus
}
#endif

#endif
