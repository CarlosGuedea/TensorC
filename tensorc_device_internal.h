/*
 * tensorc_device_internal.h — interfaz interna entre ndtensor.c y los
 * backends CUDA/Metal. El backend no conoce la estructura tc_tensor_nd ni el
 * grafo de autodiferenciación: trabaja sobre buffers crudos, formas y
 * strides. Toda la lógica de memoria perezosa (flags de validez por copia)
 * vive en ndtensor.c.
 *
 * Este header es C11 puro; no incluye cabeceras de CUDA.
 */
#ifndef TENSORC_DEVICE_INTERNAL_H
#define TENSORC_DEVICE_INTERNAL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Rango máximo de tensor que admiten los kernels (coincide con el límite
 * de perm[] en ndtensor.c). */
#define TC_DEVICE_MAX_DIMS 8

/*
 * Descriptor de broadcasting para los kernels elemento a elemento. Se pasa
 * al kernel por valor; replica la aritmética de broadcast_index() de
 * ndtensor.c: el índice lineal de salida se descompone con la forma y
 * strides de la salida y se reensambla en cada entrada saltando las
 * dimensiones de tamaño 1 (y los ejes líderes que la entrada no tiene).
 */
typedef struct {
    int ndim;   /* rango de la salida */
    int a_ndim;
    int b_ndim;
    size_t out_shape[TC_DEVICE_MAX_DIMS];
    size_t out_strides[TC_DEVICE_MAX_DIMS];
    size_t a_shape[TC_DEVICE_MAX_DIMS];
    size_t a_strides[TC_DEVICE_MAX_DIMS];
    size_t b_shape[TC_DEVICE_MAX_DIMS];
    size_t b_strides[TC_DEVICE_MAX_DIMS];
} tc_device_broadcast_desc;

/* Control de dispositivo. Son también la API pública (ndtensor.h); aquí se
 * repiten para que el backend sea autosuficiente al compilar. */
int tc__device_available(void);
int tc_cuda_available(void);
int tc_cuda_set_device(int device);
int tc_cuda_device(void);

/*
 * Umbral de elementos a partir del cual las ops elemento a elemento usan
 * GPU. Por defecto 16384; la variable de entorno específica del backend lo
 * ajusta (0 fuerza GPU en todo: así la suite ctest ejercita las rutas
 * con tensores pequeños y las valida contra diferencias finitas).
 */
size_t tc__device_min_elems(void);

/*
 * Umbral de trabajo (m·n·k) para llevar un GEMM a GPU. Producción usa
 * 4096 (mismo criterio que la ruta Accelerate); si el umbral se fija por
 * debajo, manda el valor de la variable (0 fuerza GPU incluso en GEMM pequeñas).
 */
size_t tc__device_min_gemm_work(void);

/* Memoria de dispositivo. Las copias devuelven 1 en éxito y 0 en fallo;
 * el llamador (ndtensor.c) decide si cae al camino de CPU. La memoria va
 * a un pool por tamaño exacto: los bloques liberados se reutilizan sin
 * llamar al driver (cudaMalloc/cudaFree sincronizan el dispositivo), y la
 * reutilización es segura porque todo el trabajo va al mismo stream
 * (las escrituras nuevas se encolan tras las lecturas viejas). */
float *tc__device_alloc(size_t count);
void tc__device_free(float *ptr, size_t count);
int tc__device_copy_h2d(float *dst_dev, const float *src_host, size_t count);
int tc__device_copy_d2h(float *dst_host, const float *src_dev, size_t count);

/*
 * GEMM vía cuBLAS sobre buffers row-major (el layout de TensorC). cuBLAS
 * es column-major: una matriz row-major [r,c] se interpreta como su
 * traspuesta column-major [c,r], así que cada producto se reordena como
 * Cᵀ = (A·B)ᵀ = Bᵀ·Aᵀ sin copiar datos.
 *   nn: C  = A·B        (C[m,n], A[m,k], B[k,n]; sobrescribe C)
 *   nt: dA += G·Bᵀ      (dA[m,k], G[m,n], B[k,n]; acumula con beta=1)
 *   tn: dB += Aᵀ·G      (dB[k,n], A[m,k], G[m,n]; acumula con beta=1)
 */
int tc__device_matmul_nn(const float *a, const float *b, float *c,
                       size_t m, size_t k, size_t n);
int tc__device_matmul_nt(const float *g, const float *b, float *da,
                       size_t m, size_t k, size_t n);
int tc__device_matmul_tn(const float *a, const float *g, float *db,
                       size_t m, size_t k, size_t n);

/* Metal-specific image ops; CUDA uses the CPU/Accelerate implementation. */
int tc__device_conv2d_forward(const float *x, const float *w, float *y,
                              size_t batch, size_t channels, size_t height,
                              size_t width, size_t filters, size_t kernel_h,
                              size_t kernel_w, size_t out_h, size_t out_w,
                              size_t stride_h, size_t stride_w,
                              size_t pad_h, size_t pad_w);
int tc__device_conv2d_backward(const float *x, const float *w, const float *gy,
                               float *gx, float *gw, size_t batch,
                               size_t channels, size_t height, size_t width,
                               size_t filters, size_t kernel_h,
                               size_t kernel_w, size_t out_h, size_t out_w,
                               size_t stride_h, size_t stride_w,
                               size_t pad_h, size_t pad_w, int want_x,
                               int want_w);
int tc__device_maxpool2d_forward(const float *x, float *y, size_t batch,
                                 size_t channels, size_t height,
                                 size_t width, size_t pool_h,
                                 size_t pool_w);
int tc__device_maxpool2d_backward(const float *x, const float *gy, float *gx,
                                  size_t batch, size_t channels,
                                  size_t height, size_t width,
                                  size_t pool_h, size_t pool_w);

/* Elementwise con broadcasting. Los backward ACUMULAN sobre ga/gb (la
 * semántica del grafo es de acumulación de gradientes). */
int tc__device_add_forward(const float *a, const float *b, float *out,
                         size_t n, const tc_device_broadcast_desc *desc);
int tc__device_mul_forward(const float *a, const float *b, float *out,
                         size_t n, const tc_device_broadcast_desc *desc);
int tc__device_add_backward(const float *g, float *ga, float *gb, size_t n,
                          const tc_device_broadcast_desc *desc,
                          int want_a, int want_b);
int tc__device_mul_backward(const float *g, const float *a, const float *b,
                          float *ga, float *gb, size_t n,
                          const tc_device_broadcast_desc *desc,
                          int want_a, int want_b);

/* Activaciones. Los backward de sigmoid/tanh reciben la SALIDA y (se
 * reutiliza la fórmula y·(1−y) / 1−y²) y el de relu la entrada x. */
int tc__device_relu_forward(const float *x, float *out, size_t n);
int tc__device_relu_backward(const float *x, const float *g, float *gx,
                           size_t n);
int tc__device_sigmoid_forward(const float *x, float *out, size_t n);
int tc__device_sigmoid_backward(const float *y, const float *g, float *gx,
                              size_t n);
int tc__device_tanh_forward(const float *x, float *out, size_t n);
int tc__device_tanh_backward(const float *y, const float *g, float *gx,
                           size_t n);

/* Softmax sobre la última dimensión: `rows` filas de `classes` elementos.
 * El backward acumula sobre gx. */
int tc__device_softmax_forward(const float *x, float *out, size_t rows,
                             size_t classes);
int tc__device_softmax_backward(const float *y, const float *g, float *gx,
                              size_t rows, size_t classes);

/* Layer normalization sobre la última dimensión (`rows` filas de `width`
 * elementos), misma fórmula en dos pasadas que el camino de CPU. El
 * backward acumula sobre gx. */
int tc__device_layer_norm_forward(const float *x, float *out, size_t rows,
                                size_t width);
int tc__device_layer_norm_backward(const float *x, const float *g, float *gx,
                                 size_t rows, size_t width);

/* RMSNorm sobre la última dimensión (sin resta de media, eps 1e-6), mismo
 * orden de operaciones que el camino de CPU. El backward acumula sobre gx. */
int tc__device_rms_norm_forward(const float *x, float *out, size_t rows,
                              size_t width);
int tc__device_rms_norm_backward(const float *x, const float *g, float *gx,
                               size_t rows, size_t width);

/*
 * Atención de producto punto escalado sobre lotes [batch, seq, depth]:
 * softmax((Q Kᵀ)/sqrt(depth) [+ máscara causal]) V. Las GEMM internas son
 * batidas con strides vía cuBLAS; el softmax causal es un kernel propio.
 * El backward ACUMULA sobre dq/dk/dv (punteros NULL = no se pide ese
 * gradiente) y recomputa P igual que el camino de CPU.
 */
int tc__device_attention_forward(const float *q, const float *k,
                               const float *v, float *out, size_t batch,
                               size_t q_seq, size_t k_seq, size_t depth,
                               size_t v_depth, int causal);
int tc__device_attention_backward(const float *q, const float *k,
                                const float *v, const float *dout,
                                float *dq, float *dk, float *dv,
                                size_t batch, size_t q_seq, size_t k_seq,
                                size_t depth, size_t v_depth, int causal);

/*
 * Descriptor de permutación de ejes (réplica del mapeo de
 * tc_tensor_nd_permute): el índice lineal de entrada se descompone con
 * in_shape/in_strides y se reensambla con out_strides vía axes.
 */
typedef struct {
    int ndim;
    size_t in_shape[TC_DEVICE_MAX_DIMS];
    size_t in_strides[TC_DEVICE_MAX_DIMS];
    size_t out_strides[TC_DEVICE_MAX_DIMS];
    size_t axes[TC_DEVICE_MAX_DIMS];
} tc_device_permute_desc;

/* Permute/transpose (transpose usa axes = {1, 0}). El backward acumula
 * sobre gx; la biyección garantiza un único escritor por elemento. */
int tc__device_permute_forward(const float *in, float *out, size_t n,
                             const tc_device_permute_desc *desc);
int tc__device_permute_backward(const float *g, float *gx, size_t n,
                              const tc_device_permute_desc *desc);

/* Copia device→device (reshape forward) y acumulación elemento a elemento
 * gx[i] += g[i] (backward de reshape). */
int tc__device_copy_d2d(float *dst_dev, const float *src_dev, size_t count);
int tc__device_accumulate(float *dst, const float *src, size_t n);

/* Pone a cero un buffer de device (zero_grad sin bajar a host). */
int tc__device_zero_fill(float *dst, size_t n);

/*
 * Cross-entropy con softmax ya aplicada: loss = −Σ t·log(p) / rows.
 * El forward es una reducción (dos niveles: bloque + atomicAdd escalar);
 * el backward acumula sobre gp/gt (NULL = no se pide). p y t son [rows,
 * classes] aplanados (n = rows·classes).
 */
int tc__device_cross_entropy_forward(const float *p, const float *t,
                                   float *out, size_t n, size_t rows);
int tc__device_cross_entropy_backward(const float *p, const float *t,
                                    const float *g, float *gp, float *gt,
                                    size_t n, size_t rows, int want_p,
                                    int want_t);

/*
 * Embedding (gather de filas): out[i,:] = table[ids[i],:] para n índices
 * con vectores de `dim` floats. El backward es scatter-add sobre las filas
 * de la tabla (atómico: filas repetidas acumulan) y ACUMULA sobre gtable.
 */
int tc__device_embedding_forward(const float *table, const float *ids,
                               float *out, size_t n, size_t dim);
int tc__device_embedding_backward(const float *g, const float *ids,
                                float *gtable, size_t n, size_t dim);

/*
 * Cross-entropy con objetivos enteros: p es [rows, classes] aplanado con
 * probabilidades, ids[r] es la clase correcta (float entero). El forward es
 * la reducción -mean log p[r, y_r] (el wrapper pone `out` a cero antes);
 * el backward ACUMULA solo sobre gp[r, y_r] con g el gradiente escalar.
 */
int tc__device_sparse_ce_forward(const float *p, const float *ids, float *out,
                               size_t rows, size_t classes);
int tc__device_sparse_ce_backward(const float *p, const float *ids,
                                const float *g, float *gp, size_t rows,
                                size_t classes);

/* Pasos de optimizador en device (in-place sobre param; Adam además
 * actualiza sus momentos m/v en device). bias1/bias2 son las correcciones
 * 1−beta^t ya calculadas en host. */
int tc__device_sgd_step(float *param, const float *grad, size_t n, float lr);
int tc__device_adam_step(float *param, const float *grad, float *m, float *v,
                       size_t n, float lr, float beta1, float beta2,
                       float bias1, float bias2, float epsilon);

#ifdef __cplusplus
}
#endif

#endif /* TENSORC_CUDA_INTERNAL_H */
