/*
 * tensorc_cuda.cu — backend CUDA de TensorC.
 *
 * Tercer backend tras el mismo switch que el genérico (loops C) y Apple
 * Accelerate: ndtensor.c decide por tamaño si una operación corre en GPU y
 * cae al camino genérico ante cualquier fallo. Este archivo solo ejecuta:
 * gestión de dispositivo, memoria, kernels y llamadas cuBLAS sobre buffers
 * crudos. La estructura tc_tensor_nd, el grafo y la política de copias
 * perezosas host↔device viven en ndtensor.c.
 *
 * Estilo C (el proyecto es C11); se compila con nvcc al configurar CMake
 * con -DTENSORC_USE_CUDA=ON. Cada wrapper público termina en
 * cudaDeviceSynchronize + comprobación de error: semántica síncrona y
 * todo-o-nada por llamada, así ndtensor.c puede marcar sus flags de
 * validez con seguridad. Es la v1 medible; la relajación a streams
 * asíncronos es un paso posterior, igual que la residencia prolongada.
 */
#include "tensorc_cuda_internal.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Estado del proceso: un solo dispositivo activo (v1).               */
/* ------------------------------------------------------------------ */

static int tc__cuda_state = 0; /* 0 = sin sondear, 1 = disponible, -1 = no */
static int tc__cuda_current_device = 0;
static cublasHandle_t tc__cublas_handle = NULL;
static size_t tc__min_elems = (size_t)16384;

static void tc__cuda_shutdown(void) {
    if (tc__cublas_handle) {
        cublasDestroy(tc__cublas_handle);
        tc__cublas_handle = NULL;
    }
}

static int tc__cuda_probe(void) {
    if (tc__cuda_state != 0) return tc__cuda_state > 0;
    int count = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    /* cudaGetDeviceCount deja el flag de error prendido cuando no hay
     * driver; se limpia para no contaminar llamadas posteriores. */
    cudaGetLastError();
    tc__cuda_state = (err == cudaSuccess && count > 0) ? 1 : -1;
    if (tc__cuda_state > 0) {
        const char *env = getenv("TC_CUDA_MIN_ELEMS");
        if (env && env[0])
            tc__min_elems = (size_t)strtoull(env, NULL, 10);
        cudaSetDevice(tc__cuda_current_device);
        atexit(tc__cuda_shutdown);
    }
    return tc__cuda_state > 0;
}

int tc_cuda_available(void) {
    return tc__cuda_probe();
}

int tc_cuda_set_device(int device) {
    if (!tc__cuda_probe()) return 0;
    int count = 0;
    cudaGetDeviceCount(&count);
    if (device < 0 || device >= count) return 0;
    if (cudaSetDevice(device) != cudaSuccess) return 0;
    tc__cuda_current_device = device;
    return 1;
}

int tc_cuda_device(void) {
    return tc__cuda_probe() ? tc__cuda_current_device : -1;
}

size_t tc__cuda_min_elems(void) {
    tc__cuda_probe();
    return tc__min_elems;
}

size_t tc__cuda_min_gemm_work(void) {
    tc__cuda_probe();
    return tc__min_elems < ((size_t)1 << 12) ? tc__min_elems
                                             : ((size_t)1 << 12);
}

/* ------------------------------------------------------------------ */
/* Memoria y copias.                                                   */
/* ------------------------------------------------------------------ */

float *tc__cuda_alloc(size_t count) {
    if (!tc__cuda_probe() || count == 0) return NULL;
    cudaSetDevice(tc__cuda_current_device);
    void *ptr = NULL;
    if (cudaMalloc(&ptr, count * sizeof(float)) != cudaSuccess) {
        cudaGetLastError();
        return NULL;
    }
    return (float *)ptr;
}

void tc__cuda_free(float *ptr) {
    if (!ptr || tc__cuda_state <= 0) return;
    cudaSetDevice(tc__cuda_current_device);
    cudaFree(ptr);
    cudaGetLastError(); /* liberar nunca tumba el camino de CPU */
}

int tc__cuda_copy_h2d(float *dst_dev, const float *src_host, size_t count) {
    if (!dst_dev || !src_host || tc__cuda_state <= 0) return 0;
    if (count == 0) return 1;
    cudaSetDevice(tc__cuda_current_device);
    return cudaMemcpy(dst_dev, src_host, count * sizeof(float),
                      cudaMemcpyHostToDevice) == cudaSuccess;
}

int tc__cuda_copy_d2h(float *dst_host, const float *src_dev, size_t count) {
    if (!dst_host || !src_dev || tc__cuda_state <= 0) return 0;
    if (count == 0) return 1;
    cudaSetDevice(tc__cuda_current_device);
    return cudaMemcpy(dst_host, src_dev, count * sizeof(float),
                      cudaMemcpyDeviceToHost) == cudaSuccess;
}

/* ------------------------------------------------------------------ */
/* Utilidades de lanzamiento.                                          */
/* ------------------------------------------------------------------ */

#define TC_CUDA_THREADS 256

static unsigned int tc__grid_for(size_t n) {
    return (unsigned int)((n + TC_CUDA_THREADS - 1) / TC_CUDA_THREADS);
}

/* Comprueba el lanzamiento y sincroniza: da semántica todo-o-nada. */
static int tc__cuda_commit(const char *what) {
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "TensorC CUDA: %s falló: %s\n", what,
                cudaGetErrorString(err));
        return 0;
    }
    return 1;
}

static cublasHandle_t tc__cublas(void) {
    if (!tc__cublas_handle) {
        cudaSetDevice(tc__cuda_current_device);
        if (cublasCreate(&tc__cublas_handle) != CUBLAS_STATUS_SUCCESS) {
            tc__cublas_handle = NULL;
            return NULL;
        }
    }
    return tc__cublas_handle;
}

/* ------------------------------------------------------------------ */
/* GEMM row-major vía cuBLAS.                                          */
/* ------------------------------------------------------------------ */

/* Límite de int para los parámetros de cuBLAS. */
static int tc__fits_int(size_t m, size_t k, size_t n) {
    const size_t int_max = (size_t)2147483647;
    return m > 0 && k > 0 && n > 0 &&
           m <= int_max && k <= int_max && n <= int_max;
}

int tc__cuda_matmul_nn(const float *a, const float *b, float *c,
                       size_t m, size_t k, size_t n) {
    if (!a || !b || !c || !tc__fits_int(m, k, n)) return 0;
    cublasHandle_t handle = tc__cublas();
    if (!handle) return 0;
    const float one = 1.0f, zero = 0.0f;
    /* Cᵀ[n,m] = Bᵀ[n,k] · Aᵀ[k,m]: ambas entradas pasan sin transponer. */
    const cublasStatus_t st = cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        (int)n, (int)m, (int)k,
        &one, b, (int)n, a, (int)k,
        &zero, c, (int)n);
    return st == CUBLAS_STATUS_SUCCESS && tc__cuda_commit("matmul_nn");
}

int tc__cuda_matmul_nt(const float *g, const float *b, float *da,
                       size_t m, size_t k, size_t n) {
    if (!g || !b || !da || !tc__fits_int(m, k, n)) return 0;
    cublasHandle_t handle = tc__cublas();
    if (!handle) return 0;
    const float one = 1.0f;
    /* dAᵀ[k,m] = B[k,n] · Gᵀ[n,m]: B entra transposta (OP_T sobre su
     * vista column-major [n,k]) y G sin transponer. Acumula (beta=1). */
    const cublasStatus_t st = cublasSgemm(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        (int)k, (int)m, (int)n,
        &one, b, (int)n, g, (int)n,
        &one, da, (int)k);
    return st == CUBLAS_STATUS_SUCCESS && tc__cuda_commit("matmul_nt");
}

int tc__cuda_matmul_tn(const float *a, const float *g, float *db,
                       size_t m, size_t k, size_t n) {
    if (!a || !g || !db || !tc__fits_int(m, k, n)) return 0;
    cublasHandle_t handle = tc__cublas();
    if (!handle) return 0;
    const float one = 1.0f;
    /* dBᵀ[n,k] = Gᵀ[n,m] · A[m,k]: G sin transponer y A transpuesta
     * (OP_T sobre su vista column-major [k,m]). Acumula (beta=1). */
    const cublasStatus_t st = cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_T,
        (int)n, (int)k, (int)m,
        &one, g, (int)n, a, (int)k,
        &one, db, (int)n);
    return st == CUBLAS_STATUS_SUCCESS && tc__cuda_commit("matmul_tn");
}

/* ------------------------------------------------------------------ */
/* Broadcasting en dispositivo (réplica de broadcast_index()).         */
/* ------------------------------------------------------------------ */

__device__ static size_t tc__broadcast_index(
    const size_t *in_shape, const size_t *in_strides, int in_ndim,
    const size_t *out_shape, const size_t *out_strides, int out_ndim,
    size_t out_index) {
    const int offset = out_ndim - in_ndim;
    size_t in_index = 0;
    for (int axis = 0; axis < in_ndim; ++axis) {
        const int out_axis = axis + offset;
        const size_t coordinate =
            (out_index / out_strides[out_axis]) % out_shape[out_axis];
        if (in_shape[axis] != 1)
            in_index += coordinate * in_strides[axis];
    }
    return in_index;
}

/* ------------------------------------------------------------------ */
/* Kernels elemento a elemento.                                        */
/* ------------------------------------------------------------------ */

__global__ static void tc__add_fwd_kernel(const float *a, const float *b,
                                          float *out, size_t n,
                                          tc_cuda_broadcast_desc d) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[tc__broadcast_index(d.a_shape, d.a_strides, d.a_ndim,
                                   d.out_shape, d.out_strides, d.ndim, i)] +
             b[tc__broadcast_index(d.b_shape, d.b_strides, d.b_ndim,
                                   d.out_shape, d.out_strides, d.ndim, i)];
}

__global__ static void tc__mul_fwd_kernel(const float *a, const float *b,
                                          float *out, size_t n,
                                          tc_cuda_broadcast_desc d) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[tc__broadcast_index(d.a_shape, d.a_strides, d.a_ndim,
                                   d.out_shape, d.out_strides, d.ndim, i)] *
             b[tc__broadcast_index(d.b_shape, d.b_strides, d.b_ndim,
                                   d.out_shape, d.out_strides, d.ndim, i)];
}

/* Reducción con difusión: cada elemento de salida suma su gradiente en la
 * posición que le corresponde del padre. atomicAdd sobre float: el orden
 * no es determinista, las diferencias son del orden del redondeo (la
 * suite valida con tolerancias de diferencias finitas). */
__global__ static void tc__add_bwd_kernel(const float *g, float *ga,
                                          float *gb, size_t n,
                                          tc_cuda_broadcast_desc d,
                                          int want_a, int want_b) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gi = g[i];
    if (want_a)
        atomicAdd(&ga[tc__broadcast_index(d.a_shape, d.a_strides, d.a_ndim,
                                          d.out_shape, d.out_strides, d.ndim,
                                          i)],
                  gi);
    if (want_b)
        atomicAdd(&gb[tc__broadcast_index(d.b_shape, d.b_strides, d.b_ndim,
                                          d.out_shape, d.out_strides, d.ndim,
                                          i)],
                  gi);
}

__global__ static void tc__mul_bwd_kernel(const float *g, const float *a,
                                          const float *b, float *ga,
                                          float *gb, size_t n,
                                          tc_cuda_broadcast_desc d,
                                          int want_a, int want_b) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const size_t ai = tc__broadcast_index(d.a_shape, d.a_strides, d.a_ndim,
                                          d.out_shape, d.out_strides, d.ndim,
                                          i);
    const size_t bi = tc__broadcast_index(d.b_shape, d.b_strides, d.b_ndim,
                                          d.out_shape, d.out_strides, d.ndim,
                                          i);
    const float gi = g[i];
    if (want_a) atomicAdd(&ga[ai], gi * b[bi]);
    if (want_b) atomicAdd(&gb[bi], gi * a[ai]);
}

__global__ static void tc__relu_fwd_kernel(const float *x, float *out,
                                           size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

__global__ static void tc__relu_bwd_kernel(const float *x, const float *g,
                                           float *gx, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    gx[i] += x[i] > 0.0f ? g[i] : 0.0f;
}

__global__ static void tc__sigmoid_fwd_kernel(const float *x, float *out,
                                              size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    /* Rama estable según el signo, igual que el camino de CPU. */
    const float v = x[i];
    out[i] = v >= 0.0f ? 1.0f / (1.0f + expf(-v))
                       : expf(v) / (1.0f + expf(v));
}

__global__ static void tc__sigmoid_bwd_kernel(const float *y, const float *g,
                                              float *gx, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    gx[i] += g[i] * y[i] * (1.0f - y[i]);
}

__global__ static void tc__tanh_fwd_kernel(const float *x, float *out,
                                           size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = tanhf(x[i]);
}

__global__ static void tc__tanh_bwd_kernel(const float *y, const float *g,
                                           float *gx, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    gx[i] += g[i] * (1.0f - y[i] * y[i]);
}

/* Softmax: un hilo por fila. La paralelización fina por fila (reducciones
 * en memoria compartida) queda como mejora posterior si el perfil lo pide. */
__global__ static void tc__softmax_fwd_kernel(const float *x, float *out,
                                              size_t rows, size_t classes) {
    const size_t r = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float *src = x + r * classes;
    float *dst = out + r * classes;
    float max_value = src[0];
    for (size_t j = 1; j < classes; ++j)
        if (src[j] > max_value) max_value = src[j];
    float sum = 0.0f;
    for (size_t j = 0; j < classes; ++j) {
        dst[j] = expf(src[j] - max_value);
        sum += dst[j];
    }
    for (size_t j = 0; j < classes; ++j) dst[j] /= sum;
}

__global__ static void tc__softmax_bwd_kernel(const float *y, const float *g,
                                              float *gx, size_t rows,
                                              size_t classes) {
    const size_t r = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float *prob = y + r * classes;
    const float *grad_out = g + r * classes;
    float *grad_in = gx + r * classes;
    float dot = 0.0f;
    for (size_t j = 0; j < classes; ++j) dot += grad_out[j] * prob[j];
    for (size_t j = 0; j < classes; ++j)
        grad_in[j] += prob[j] * (grad_out[j] - dot);
}

/* ------------------------------------------------------------------ */
/* Wrappers de los kernels.                                            */
/* ------------------------------------------------------------------ */

int tc__cuda_add_forward(const float *a, const float *b, float *out,
                         size_t n, const tc_cuda_broadcast_desc *desc) {
    if (!a || !b || !out || !desc || tc__cuda_state <= 0) return 0;
    if (n == 0) return 1;
    cudaSetDevice(tc__cuda_current_device);
    tc__add_fwd_kernel<<<tc__grid_for(n), TC_CUDA_THREADS>>>(a, b, out, n,
                                                             *desc);
    return tc__cuda_commit("add_forward");
}

int tc__cuda_mul_forward(const float *a, const float *b, float *out,
                         size_t n, const tc_cuda_broadcast_desc *desc) {
    if (!a || !b || !out || !desc || tc__cuda_state <= 0) return 0;
    if (n == 0) return 1;
    cudaSetDevice(tc__cuda_current_device);
    tc__mul_fwd_kernel<<<tc__grid_for(n), TC_CUDA_THREADS>>>(a, b, out, n,
                                                             *desc);
    return tc__cuda_commit("mul_forward");
}

int tc__cuda_add_backward(const float *g, float *ga, float *gb, size_t n,
                          const tc_cuda_broadcast_desc *desc,
                          int want_a, int want_b) {
    if (!g || !desc || tc__cuda_state <= 0) return 0;
    if (want_a && !ga) return 0;
    if (want_b && !gb) return 0;
    if (!want_a && !want_b) return 1;
    if (n == 0) return 1;
    cudaSetDevice(tc__cuda_current_device);
    tc__add_bwd_kernel<<<tc__grid_for(n), TC_CUDA_THREADS>>>(
        g, ga, gb, n, *desc, want_a, want_b);
    return tc__cuda_commit("add_backward");
}

int tc__cuda_mul_backward(const float *g, const float *a, const float *b,
                          float *ga, float *gb, size_t n,
                          const tc_cuda_broadcast_desc *desc,
                          int want_a, int want_b) {
    if (!g || !a || !b || !desc || tc__cuda_state <= 0) return 0;
    if (want_a && !ga) return 0;
    if (want_b && !gb) return 0;
    if (!want_a && !want_b) return 1;
    if (n == 0) return 1;
    cudaSetDevice(tc__cuda_current_device);
    tc__mul_bwd_kernel<<<tc__grid_for(n), TC_CUDA_THREADS>>>(
        g, a, b, ga, gb, n, *desc, want_a, want_b);
    return tc__cuda_commit("mul_backward");
}

int tc__cuda_relu_forward(const float *x, float *out, size_t n) {
    if (!x || !out || tc__cuda_state <= 0) return 0;
    if (n == 0) return 1;
    cudaSetDevice(tc__cuda_current_device);
    tc__relu_fwd_kernel<<<tc__grid_for(n), TC_CUDA_THREADS>>>(x, out, n);
    return tc__cuda_commit("relu_forward");
}

int tc__cuda_relu_backward(const float *x, const float *g, float *gx,
                           size_t n) {
    if (!x || !g || !gx || tc__cuda_state <= 0) return 0;
    if (n == 0) return 1;
    cudaSetDevice(tc__cuda_current_device);
    tc__relu_bwd_kernel<<<tc__grid_for(n), TC_CUDA_THREADS>>>(x, g, gx, n);
    return tc__cuda_commit("relu_backward");
}

int tc__cuda_sigmoid_forward(const float *x, float *out, size_t n) {
    if (!x || !out || tc__cuda_state <= 0) return 0;
    if (n == 0) return 1;
    cudaSetDevice(tc__cuda_current_device);
    tc__sigmoid_fwd_kernel<<<tc__grid_for(n), TC_CUDA_THREADS>>>(x, out, n);
    return tc__cuda_commit("sigmoid_forward");
}

int tc__cuda_sigmoid_backward(const float *y, const float *g, float *gx,
                              size_t n) {
    if (!y || !g || !gx || tc__cuda_state <= 0) return 0;
    if (n == 0) return 1;
    cudaSetDevice(tc__cuda_current_device);
    tc__sigmoid_bwd_kernel<<<tc__grid_for(n), TC_CUDA_THREADS>>>(y, g, gx, n);
    return tc__cuda_commit("sigmoid_backward");
}

int tc__cuda_tanh_forward(const float *x, float *out, size_t n) {
    if (!x || !out || tc__cuda_state <= 0) return 0;
    if (n == 0) return 1;
    cudaSetDevice(tc__cuda_current_device);
    tc__tanh_fwd_kernel<<<tc__grid_for(n), TC_CUDA_THREADS>>>(x, out, n);
    return tc__cuda_commit("tanh_forward");
}

int tc__cuda_tanh_backward(const float *y, const float *g, float *gx,
                           size_t n) {
    if (!y || !g || !gx || tc__cuda_state <= 0) return 0;
    if (n == 0) return 1;
    cudaSetDevice(tc__cuda_current_device);
    tc__tanh_bwd_kernel<<<tc__grid_for(n), TC_CUDA_THREADS>>>(y, g, gx, n);
    return tc__cuda_commit("tanh_backward");
}

int tc__cuda_softmax_forward(const float *x, float *out, size_t rows,
                             size_t classes) {
    if (!x || !out || classes == 0 || tc__cuda_state <= 0) return 0;
    if (rows == 0) return 1;
    cudaSetDevice(tc__cuda_current_device);
    tc__softmax_fwd_kernel<<<tc__grid_for(rows), TC_CUDA_THREADS>>>(
        x, out, rows, classes);
    return tc__cuda_commit("softmax_forward");
}

int tc__cuda_softmax_backward(const float *y, const float *g, float *gx,
                              size_t rows, size_t classes) {
    if (!y || !g || !gx || classes == 0 || tc__cuda_state <= 0) return 0;
    if (rows == 0) return 1;
    cudaSetDevice(tc__cuda_current_device);
    tc__softmax_bwd_kernel<<<tc__grid_for(rows), TC_CUDA_THREADS>>>(
        y, g, gx, rows, classes);
    return tc__cuda_commit("softmax_backward");
}

/* ------------------------------------------------------------------ */
/* GEMM batida con strides (atención).                                 */
/* ------------------------------------------------------------------ */

/*
 * Envoltura directa de cublasSgemmStridedBatched. Todas las matrices de
 * la atención son row-major contiguas por lote, así que se aplican las
 * mismas identidades de transposición que en los wrappers simples:
 * una matriz row-major [r,c] se lee como su traspuesta column-major.
 */
static int tc__gemm_strided_batched(cublasOperation_t ta,
                                    cublasOperation_t tb, int m, int n,
                                    int k, float alpha, const float *a,
                                    int lda, long long sa, const float *b,
                                    int ldb, long long sb, float beta,
                                    float *c, int ldc, long long sc,
                                    int batch) {
    cublasHandle_t handle = tc__cublas();
    if (!handle) return 0;
    return cublasSgemmStridedBatched(handle, ta, tb, m, n, k, &alpha, a,
                                     lda, sa, b, ldb, sb, &beta, c, ldc, sc,
                                     batch) == CUBLAS_STATUS_SUCCESS;
}

/* Caben todos los parámetros de la atención en int (requisito cuBLAS). */
static int tc__attention_fits_int(size_t batch, size_t q_seq, size_t k_seq,
                                  size_t depth, size_t v_depth) {
    const size_t int_max = (size_t)INT_MAX;
    return batch > 0 && batch <= int_max && q_seq > 0 && q_seq <= int_max &&
           k_seq > 0 && k_seq <= int_max && depth > 0 && depth <= int_max &&
           v_depth > 0 && v_depth <= int_max;
}

/* ------------------------------------------------------------------ */
/* Kernels de layer normalization (un hilo por fila, como softmax).    */
/* Réplica exacta del orden de operaciones del camino de CPU.          */
/* ------------------------------------------------------------------ */

__global__ static void tc__layer_norm_fwd_kernel(const float *x, float *out,
                                                 size_t rows, size_t width) {
    const size_t r = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float *src = x + r * width;
    float *dst = out + r * width;
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

__global__ static void tc__layer_norm_bwd_kernel(const float *x,
                                                 const float *g, float *gx,
                                                 size_t rows, size_t width) {
    const size_t r = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float *src = x + r * width;
    const float *dy = g + r * width;
    float *dx = gx + r * width;
    float mean = 0.0f, variance = 0.0f;
    for (size_t j = 0; j < width; ++j) mean += src[j];
    mean /= (float)width;
    for (size_t j = 0; j < width; ++j) {
        const float d = src[j] - mean;
        variance += d * d;
    }
    const float inv_std = 1.0f / sqrtf(variance / (float)width + 1e-5f);
    float mean_dy = 0.0f, mean_dy_xhat = 0.0f;
    for (size_t j = 0; j < width; ++j) {
        mean_dy += dy[j];
        mean_dy_xhat += dy[j] * (src[j] - mean) * inv_std;
    }
    mean_dy /= (float)width;
    mean_dy_xhat /= (float)width;
    for (size_t j = 0; j < width; ++j) {
        const float xhat = (src[j] - mean) * inv_std;
        dx[j] += inv_std * (dy[j] - mean_dy - xhat * mean_dy_xhat);
    }
}

/* ------------------------------------------------------------------ */
/* Kernels de atención: softmax (con máscara causal opcional) sobre las */
/* filas de la matriz de scores, y el paso dS del backward.            */
/* ------------------------------------------------------------------ */

/* In-place sobre scores [batch, q_seq, k_seq] aplanado a filas de k_seq.
 * La máscara causal usa el índice de fila dentro del lote (i = r % q_seq)
 * y pone -inf en j > i, igual que el camino de CPU. */
__global__ static void tc__attn_softmax_kernel(float *scores, size_t rows,
                                               size_t q_seq, size_t cols,
                                               int causal) {
    const size_t r = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    float *row = scores + r * cols;
    if (causal) {
        const size_t i = r % q_seq;
        for (size_t j = i + 1; j < cols; ++j) row[j] = -INFINITY;
    }
    float max_value = row[0];
    for (size_t j = 1; j < cols; ++j)
        if (row[j] > max_value) max_value = row[j];
    float sum = 0.0f;
    for (size_t j = 0; j < cols; ++j) {
        row[j] = expf(row[j] - max_value);
        sum += row[j];
    }
    for (size_t j = 0; j < cols; ++j) row[j] /= sum;
}

/* dS = P ⊙ (dP − rowsum(dP ⊙ P)) · scale, in-place sobre dp. */
__global__ static void tc__attn_ds_kernel(const float *p, float *dp,
                                          size_t rows, size_t cols,
                                          float scale) {
    const size_t r = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float *pr = p + r * cols;
    float *dr = dp + r * cols;
    float row_dot = 0.0f;
    for (size_t j = 0; j < cols; ++j) row_dot += dr[j] * pr[j];
    for (size_t j = 0; j < cols; ++j)
        dr[j] = pr[j] * (dr[j] - row_dot) * scale;
}

/* ------------------------------------------------------------------ */
/* Atención: forward y backward vía GEMM batida + kernels.             */
/* ------------------------------------------------------------------ */

int tc__cuda_attention_forward(const float *q, const float *k,
                               const float *v, float *out, size_t batch,
                               size_t q_seq, size_t k_seq, size_t depth,
                               size_t v_depth, int causal) {
    if (!q || !k || !v || !out || tc__cuda_state <= 0) return 0;
    if (!tc__attention_fits_int(batch, q_seq, k_seq, depth, v_depth))
        return 0;
    const size_t score_count = q_seq * k_seq;
    if (batch > ((size_t)-1) / score_count) return 0;
    float *scores = tc__cuda_alloc(batch * score_count);
    if (!scores) return 0;
    const float scale = 1.0f / sqrtf((float)depth);
    cudaSetDevice(tc__cuda_current_device);
    /* S = scale · Q·Kᵀ por lote: Sᵀ[k_seq,q_seq] = K·Qᵀ se escribe con
     * K transpuesta (OP_T) y Q sin transponer; alpha lleva la escala. */
    int ok = tc__gemm_strided_batched(
        CUBLAS_OP_T, CUBLAS_OP_N, (int)k_seq, (int)q_seq, (int)depth, scale,
        k, (int)depth, (long long)(k_seq * depth),
        q, (int)depth, (long long)(q_seq * depth),
        0.0f, scores, (int)k_seq, (long long)score_count, (int)batch);
    if (ok) {
        tc__attn_softmax_kernel<<<tc__grid_for(batch * q_seq),
                                  TC_CUDA_THREADS>>>(
            scores, batch * q_seq, q_seq, k_seq, causal);
        /* out = P·V por lote (misma forma que matmul_nn). */
        ok = tc__gemm_strided_batched(
            CUBLAS_OP_N, CUBLAS_OP_N, (int)v_depth, (int)q_seq, (int)k_seq,
            1.0f, v, (int)v_depth, (long long)(k_seq * v_depth),
            scores, (int)k_seq, (long long)score_count,
            0.0f, out, (int)v_depth, (long long)(q_seq * v_depth),
            (int)batch);
    }
    tc__cuda_free(scores);
    return ok && tc__cuda_commit("attention_forward");
}

/* Fallo a mitad de la acumulación de gradientes = estado irrecuperable
 * (misma política que el backward de matmul en ndtensor.c). */
static void tc__attention_fatal(const char *what) {
    fprintf(stderr,
            "TensorC CUDA: %s falló tras acumular gradientes parciales de "
            "atención; estado irrecuperable.\n",
            what);
    abort();
}

int tc__cuda_attention_backward(const float *q, const float *k,
                                const float *v, const float *dout,
                                float *dq, float *dk, float *dv,
                                size_t batch, size_t q_seq, size_t k_seq,
                                size_t depth, size_t v_depth, int causal) {
    if (!q || !k || !v || !dout || tc__cuda_state <= 0) return 0;
    if (!dq && !dk && !dv) return 1;
    if (!tc__attention_fits_int(batch, q_seq, k_seq, depth, v_depth))
        return 0;
    const size_t score_count = q_seq * k_seq;
    if (batch > ((size_t)-1) / score_count) return 0;
    float *p = tc__cuda_alloc(batch * score_count);
    float *dp = tc__cuda_alloc(batch * score_count);
    if (!p || !dp) {
        tc__cuda_free(p);
        tc__cuda_free(dp);
        return 0;
    }
    const float scale = 1.0f / sqrtf((float)depth);
    const size_t rows = batch * q_seq;
    cudaSetDevice(tc__cuda_current_device);
    /* Rehacer P (idéntico al forward, sin escribir la salida). */
    int ok = tc__gemm_strided_batched(
        CUBLAS_OP_T, CUBLAS_OP_N, (int)k_seq, (int)q_seq, (int)depth, scale,
        k, (int)depth, (long long)(k_seq * depth),
        q, (int)depth, (long long)(q_seq * depth),
        0.0f, p, (int)k_seq, (long long)score_count, (int)batch);
    if (ok) {
        tc__attn_softmax_kernel<<<tc__grid_for(rows), TC_CUDA_THREADS>>>(
            p, rows, q_seq, k_seq, causal);
        /* dP = dO·Vᵀ (misma forma que los scores, con dO y V). */
        ok = tc__gemm_strided_batched(
            CUBLAS_OP_T, CUBLAS_OP_N, (int)k_seq, (int)q_seq, (int)v_depth,
            1.0f, v, (int)v_depth, (long long)(k_seq * v_depth),
            dout, (int)v_depth, (long long)(q_seq * v_depth),
            0.0f, dp, (int)k_seq, (long long)score_count, (int)batch);
    }
    if (ok)
        tc__attn_ds_kernel<<<tc__grid_for(rows), TC_CUDA_THREADS>>>(
            p, dp, rows, k_seq, scale);
    if (!ok || !tc__cuda_commit("attention_backward prep")) {
        /* Nada se acumuló todavía: el llamador puede caer a CPU. */
        tc__cuda_free(p);
        tc__cuda_free(dp);
        return 0;
    }
    /* Acumulaciones: dV += Pᵀ·dO, dQ += dS·K, dK += dSᵀ·Q (beta=1). Un
     * fallo aquí dejaría gradientes a medias: se aborta con diagnóstico. */
    if (dv && !tc__gemm_strided_batched(
                  CUBLAS_OP_N, CUBLAS_OP_T, (int)v_depth, (int)k_seq,
                  (int)q_seq, 1.0f,
                  dout, (int)v_depth, (long long)(q_seq * v_depth),
                  p, (int)k_seq, (long long)score_count,
                  1.0f, dv, (int)v_depth, (long long)(k_seq * v_depth),
                  (int)batch))
        tc__attention_fatal("dV");
    if (dq && !tc__gemm_strided_batched(
                  CUBLAS_OP_N, CUBLAS_OP_N, (int)depth, (int)q_seq,
                  (int)k_seq, 1.0f,
                  k, (int)depth, (long long)(k_seq * depth),
                  dp, (int)k_seq, (long long)score_count,
                  1.0f, dq, (int)depth, (long long)(q_seq * depth),
                  (int)batch))
        tc__attention_fatal("dQ");
    if (dk && !tc__gemm_strided_batched(
                  CUBLAS_OP_N, CUBLAS_OP_T, (int)depth, (int)k_seq,
                  (int)q_seq, 1.0f,
                  q, (int)depth, (long long)(q_seq * depth),
                  dp, (int)k_seq, (long long)score_count,
                  1.0f, dk, (int)depth, (long long)(k_seq * depth),
                  (int)batch))
        tc__attention_fatal("dK");
    tc__cuda_free(p);
    tc__cuda_free(dp);
    return tc__cuda_commit("attention_backward");
}

/* ------------------------------------------------------------------ */
/* Permute / transpose (biyección por índices) y reshape.              */
/* ------------------------------------------------------------------ */

__global__ static void tc__permute_fwd_kernel(const float *in, float *out,
                                              size_t n,
                                              tc_cuda_permute_desc d) {
    const size_t lin = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (lin >= n) return;
    size_t in_idx[TC_CUDA_MAX_DIMS];
    for (int dim = 0; dim < d.ndim; ++dim)
        in_idx[dim] = (lin / d.in_strides[dim]) % d.in_shape[dim];
    size_t out_lin = 0;
    for (int k = 0; k < d.ndim; ++k)
        out_lin += in_idx[d.axes[k]] * d.out_strides[k];
    out[out_lin] = in[lin];
}

__global__ static void tc__permute_bwd_kernel(const float *g, float *gx,
                                              size_t n,
                                              tc_cuda_permute_desc d) {
    const size_t lin = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (lin >= n) return;
    size_t in_idx[TC_CUDA_MAX_DIMS];
    for (int dim = 0; dim < d.ndim; ++dim)
        in_idx[dim] = (lin / d.in_strides[dim]) % d.in_shape[dim];
    size_t out_lin = 0;
    for (int k = 0; k < d.ndim; ++k)
        out_lin += in_idx[d.axes[k]] * d.out_strides[k];
    gx[lin] += g[out_lin];
}

__global__ static void tc__accumulate_kernel(float *dst, const float *src,
                                             size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] += src[i];
}

int tc__cuda_permute_forward(const float *in, float *out, size_t n,
                             const tc_cuda_permute_desc *desc) {
    if (!in || !out || !desc || tc__cuda_state <= 0) return 0;
    if (n == 0) return 1;
    cudaSetDevice(tc__cuda_current_device);
    tc__permute_fwd_kernel<<<tc__grid_for(n), TC_CUDA_THREADS>>>(in, out, n,
                                                                 *desc);
    return tc__cuda_commit("permute_forward");
}

int tc__cuda_permute_backward(const float *g, float *gx, size_t n,
                              const tc_cuda_permute_desc *desc) {
    if (!g || !gx || !desc || tc__cuda_state <= 0) return 0;
    if (n == 0) return 1;
    cudaSetDevice(tc__cuda_current_device);
    tc__permute_bwd_kernel<<<tc__grid_for(n), TC_CUDA_THREADS>>>(g, gx, n,
                                                                 *desc);
    return tc__cuda_commit("permute_backward");
}

int tc__cuda_copy_d2d(float *dst_dev, const float *src_dev, size_t count) {
    if (!dst_dev || !src_dev || tc__cuda_state <= 0) return 0;
    if (count == 0) return 1;
    cudaSetDevice(tc__cuda_current_device);
    return cudaMemcpy(dst_dev, src_dev, count * sizeof(float),
                      cudaMemcpyDeviceToDevice) == cudaSuccess;
}

int tc__cuda_accumulate(float *dst, const float *src, size_t n) {
    if (!dst || !src || tc__cuda_state <= 0) return 0;
    if (n == 0) return 1;
    cudaSetDevice(tc__cuda_current_device);
    tc__accumulate_kernel<<<tc__grid_for(n), TC_CUDA_THREADS>>>(dst, src, n);
    return tc__cuda_commit("accumulate");
}

/* ------------------------------------------------------------------ */
/* Wrappers de layer normalization.                                    */
/* ------------------------------------------------------------------ */

int tc__cuda_layer_norm_forward(const float *x, float *out, size_t rows,
                                size_t width) {
    if (!x || !out || width == 0 || tc__cuda_state <= 0) return 0;
    if (rows == 0) return 1;
    cudaSetDevice(tc__cuda_current_device);
    tc__layer_norm_fwd_kernel<<<tc__grid_for(rows), TC_CUDA_THREADS>>>(
        x, out, rows, width);
    return tc__cuda_commit("layer_norm_forward");
}

int tc__cuda_layer_norm_backward(const float *x, const float *g, float *gx,
                                 size_t rows, size_t width) {
    if (!x || !g || !gx || width == 0 || tc__cuda_state <= 0) return 0;
    if (rows == 0) return 1;
    cudaSetDevice(tc__cuda_current_device);
    tc__layer_norm_bwd_kernel<<<tc__grid_for(rows), TC_CUDA_THREADS>>>(
        x, g, gx, rows, width);
    return tc__cuda_commit("layer_norm_backward");
}
