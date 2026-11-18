/*
 * tensorc_cuda_internal.h — interfaz interna entre ndtensor.c y el backend
 * CUDA (tensorc_cuda.cu). Solo se usa cuando ndtensor.c se compila con
 * TENSORC_HAS_CUDA. El backend no conoce la estructura tc_tensor_nd ni el
 * grafo de autodiferenciación: trabaja sobre buffers crudos, formas y
 * strides. Toda la lógica de memoria perezosa (flags de validez por copia)
 * vive en ndtensor.c.
 *
 * Este header es C11 puro; no incluye cabeceras de CUDA.
 */
#ifndef TENSORC_CUDA_INTERNAL_H
#define TENSORC_CUDA_INTERNAL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Rango máximo de tensor que admiten los kernels (coincide con el límite
 * de perm[] en ndtensor.c). */
#define TC_CUDA_MAX_DIMS 8

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
    size_t out_shape[TC_CUDA_MAX_DIMS];
    size_t out_strides[TC_CUDA_MAX_DIMS];
    size_t a_shape[TC_CUDA_MAX_DIMS];
    size_t a_strides[TC_CUDA_MAX_DIMS];
    size_t b_shape[TC_CUDA_MAX_DIMS];
    size_t b_strides[TC_CUDA_MAX_DIMS];
} tc_cuda_broadcast_desc;

/* Control de dispositivo. Son también la API pública (ndtensor.h); aquí se
 * repiten para que el backend sea autosuficiente al compilar. */
int tc_cuda_available(void);
int tc_cuda_set_device(int device);
int tc_cuda_device(void);

/*
 * Umbral de elementos a partir del cual las ops elemento a elemento usan
 * GPU. Por defecto 16384; la variable de entorno TC_CUDA_MIN_ELEMS lo
 * ajusta (0 fuerza GPU en todo: así la suite ctest ejercita las rutas CUDA
 * con tensores pequeños y las valida contra diferencias finitas).
 */
size_t tc__cuda_min_elems(void);

/*
 * Umbral de trabajo (m·n·k) para llevar un GEMM a GPU. Producción usa
 * 4096 (mismo criterio que la ruta Accelerate); si TC_CUDA_MIN_ELEMS se
 * fija por debajo, manda el valor de la variable (0 = forzar GPU siempre,
 * para que la suite valide las GEMM de cuBLAS con matrices pequeñas).
 */
size_t tc__cuda_min_gemm_work(void);

/* Memoria de dispositivo. Las copias devuelven 1 en éxito y 0 en fallo;
 * el llamador (ndtensor.c) decide si cae al camino de CPU. */
float *tc__cuda_alloc(size_t count);
void tc__cuda_free(float *ptr);
int tc__cuda_copy_h2d(float *dst_dev, const float *src_host, size_t count);
int tc__cuda_copy_d2h(float *dst_host, const float *src_dev, size_t count);

/*
 * GEMM vía cuBLAS sobre buffers row-major (el layout de TensorC). cuBLAS
 * es column-major: una matriz row-major [r,c] se interpreta como su
 * traspuesta column-major [c,r], así que cada producto se reordena como
 * Cᵀ = (A·B)ᵀ = Bᵀ·Aᵀ sin copiar datos.
 *   nn: C  = A·B        (C[m,n], A[m,k], B[k,n]; sobrescribe C)
 *   nt: dA += G·Bᵀ      (dA[m,k], G[m,n], B[k,n]; acumula con beta=1)
 *   tn: dB += Aᵀ·G      (dB[k,n], A[m,k], G[m,n]; acumula con beta=1)
 */
int tc__cuda_matmul_nn(const float *a, const float *b, float *c,
                       size_t m, size_t k, size_t n);
int tc__cuda_matmul_nt(const float *g, const float *b, float *da,
                       size_t m, size_t k, size_t n);
int tc__cuda_matmul_tn(const float *a, const float *g, float *db,
                       size_t m, size_t k, size_t n);

/* Elementwise con broadcasting. Los backward ACUMULAN sobre ga/gb (la
 * semántica del grafo es de acumulación de gradientes). */
int tc__cuda_add_forward(const float *a, const float *b, float *out,
                         size_t n, const tc_cuda_broadcast_desc *desc);
int tc__cuda_mul_forward(const float *a, const float *b, float *out,
                         size_t n, const tc_cuda_broadcast_desc *desc);
int tc__cuda_add_backward(const float *g, float *ga, float *gb, size_t n,
                          const tc_cuda_broadcast_desc *desc,
                          int want_a, int want_b);
int tc__cuda_mul_backward(const float *g, const float *a, const float *b,
                          float *ga, float *gb, size_t n,
                          const tc_cuda_broadcast_desc *desc,
                          int want_a, int want_b);

/* Activaciones. Los backward de sigmoid/tanh reciben la SALIDA y (se
 * reutiliza la fórmula y·(1−y) / 1−y²) y el de relu la entrada x. */
int tc__cuda_relu_forward(const float *x, float *out, size_t n);
int tc__cuda_relu_backward(const float *x, const float *g, float *gx,
                           size_t n);
int tc__cuda_sigmoid_forward(const float *x, float *out, size_t n);
int tc__cuda_sigmoid_backward(const float *y, const float *g, float *gx,
                              size_t n);
int tc__cuda_tanh_forward(const float *x, float *out, size_t n);
int tc__cuda_tanh_backward(const float *y, const float *g, float *gx,
                           size_t n);

/* Softmax sobre la última dimensión: `rows` filas de `classes` elementos.
 * El backward acumula sobre gx. */
int tc__cuda_softmax_forward(const float *x, float *out, size_t rows,
                             size_t classes);
int tc__cuda_softmax_backward(const float *y, const float *g, float *gx,
                              size_t rows, size_t classes);

#ifdef __cplusplus
}
#endif

#endif /* TENSORC_CUDA_INTERNAL_H */
