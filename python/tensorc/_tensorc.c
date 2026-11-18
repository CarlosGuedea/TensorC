/*
 * Python bindings for TensorC: a CPython extension module (C11) that wraps
 * the tc_tensor_nd autograd API. The numerical core lives entirely in
 * ndtensor.c; this file only converts between Python objects and tensors.
 *
 * Usage:
 *   import tensorc
 *   x = tensorc.Tensor([1, 2, 3, 4], shape=(2, 2), requires_grad=True)
 *   y = x.matmul(w).relu()
 *   loss = y.sum()
 *   loss.backward()
 *   x.sgd_step(0.01)
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdlib.h>
#include <string.h>

#include <tensorc/tensorc.h>

typedef struct {
    PyObject_HEAD
    tc_tensor_nd *tensor;
} TensorObject;

static PyTypeObject TensorType;

static TensorObject *tensor_wrap(tc_tensor_nd *tensor) {
    if (!tensor) {
        PyErr_SetString(PyExc_RuntimeError, "operación TensorC falló");
        return NULL;
    }
    TensorObject *obj = PyObject_New(TensorObject, &TensorType);
    if (!obj) {
        tc_tensor_nd_free(tensor);
        return NULL;
    }
    obj->tensor = tensor;
    return obj;
}

/* Extrae el tc_tensor_nd subyacente de un TensorObject. */
static tc_tensor_nd *as_tensor(PyObject *obj) {
    if (!PyObject_TypeCheck(obj, &TensorType)) {
        PyErr_SetString(PyExc_TypeError, "se esperaba un tensorc.Tensor");
        return NULL;
    }
    return ((TensorObject *)obj)->tensor;
}

static void tensor_dealloc(TensorObject *self) {
    tc_tensor_nd_free(self->tensor);
    PyObject_Del(self);
}

/* Convierte una tupla/lista de Python en shape de tama_t. */
static int parse_shape(PyObject *obj, size_t *shape, size_t *ndim) {
    PyObject *seq = PySequence_Fast(obj, "shape debe ser una secuencia");
    if (!seq) return -1;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if (n < 1 || n > 8) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "shape debe tener entre 1 y 8 dims");
        return -1;
    }
    for (Py_ssize_t i = 0; i < n; ++i) {
        const long dim = PyLong_AsLong(PySequence_Fast_GET_ITEM(seq, i));
        if (dim < 0 || PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_SetString(PyExc_ValueError, "dimensiones inválidas en shape");
            return -1;
        }
        shape[i] = (size_t)dim;
    }
    *ndim = (size_t)n;
    Py_DECREF(seq);
    return 0;
}

/* Rellena `out` desde un Py_buffer contiguo cuyo formato ya fue validado:
   'f' (float32) se copia tal cual, 'd' (float64) se convierte a float. */
static void copy_from_buffer(float *out, const Py_buffer *view) {
    const size_t count = (size_t)view->len / (size_t)view->itemsize;
    if (view->itemsize == (Py_ssize_t)sizeof(float)) {
        memcpy(out, view->buf, count * sizeof(float));
    } else {
        const double *src = (const double *)view->buf;
        for (size_t i = 0; i < count; ++i) out[i] = (float)src[i];
    }
}

static PyObject *tensor_new(PyTypeObject *type, PyObject *args,
                            PyObject *kwargs) {
    static char *names[] = {"data", "shape", "requires_grad", NULL};
    PyObject *data_obj, *shape_obj = NULL;
    int requires_grad = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|Op", names, &data_obj,
                                     &shape_obj, &requires_grad))
        return NULL;

    /* Ruta rápida: objetos con protocolo de buffer (NumPy, array, ...).
       Soporta float32 y float64, y la shape se infiere del buffer. */
    float *data = NULL;
    Py_ssize_t count = 0;
    size_t shape[8], ndim = 0;
    int have_shape = 0;
    Py_buffer view;
    if (PyObject_CheckBuffer(data_obj) &&
        PyObject_GetBuffer(data_obj, &view,
                           PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) == 0) {
        char type = '\0';
        if (view.format) {
            const char *f = view.format;
            if (*f == '@' || *f == '=' || *f == '<' || *f == '>' || *f == '!')
                ++f;
            while (*f >= '0' && *f <= '9') ++f;
            type = *f;
        }
        const int is_f32 = type == 'f' && view.itemsize == 4;
        const int is_f64 = type == 'd' && view.itemsize == 8;
        if (is_f32 || is_f64) {
            count = view.len / view.itemsize;
            if (view.ndim >= 1 && view.ndim <= 8 && view.shape) {
                ndim = (size_t)view.ndim;
                for (Py_ssize_t i = 0; i < view.ndim; ++i) {
                    if (view.shape[i] < 0) {
                        PyBuffer_Release(&view);
                        PyErr_SetString(PyExc_ValueError,
                                        "shape con dimensión negativa");
                        return NULL;
                    }
                    shape[i] = (size_t)view.shape[i];
                }
                have_shape = 1;
            }
            data = (float *)PyMem_Malloc((size_t)(count > 0 ? count : 1) *
                                         sizeof(float));
            if (!data) {
                PyBuffer_Release(&view);
                return PyErr_NoMemory();
            }
            copy_from_buffer(data, &view);
        }
        PyBuffer_Release(&view);
    }

    /* Ruta general: cualquier secuencia de números (listas, tuplas). */
    if (!data) {
        PyObject *seq =
            PySequence_Fast(data_obj, "data debe ser una secuencia o buffer");
        if (!seq) return NULL;
        count = PySequence_Fast_GET_SIZE(seq);
        data = (float *)PyMem_Malloc((size_t)(count > 0 ? count : 1) *
                                     sizeof(float));
        if (!data) {
            Py_DECREF(seq);
            return PyErr_NoMemory();
        }
        for (Py_ssize_t i = 0; i < count; ++i) {
            const float value =
                (float)PyFloat_AsDouble(PySequence_Fast_GET_ITEM(seq, i));
            if (PyErr_Occurred()) {
                PyMem_Free(data);
                Py_DECREF(seq);
                return NULL;
            }
            data[i] = value;
        }
        Py_DECREF(seq);
    }

    size_t final_shape[8], final_ndim;
    if (shape_obj) {
        if (parse_shape(shape_obj, final_shape, &final_ndim) < 0) {
            PyMem_Free(data);
            return NULL;
        }
    } else if (have_shape) {
        memcpy(final_shape, shape, ndim * sizeof(size_t));
        final_ndim = ndim;
    } else {
        final_shape[0] = (size_t)count;
        final_ndim = 1;
    }
    size_t expected = 1;
    for (size_t i = 0; i < final_ndim; ++i) expected *= final_shape[i];
    if ((Py_ssize_t)expected != count) {
        PyMem_Free(data);
        PyErr_SetString(PyExc_ValueError,
                        "el número de datos no coincide con shape");
        return NULL;
    }
    tc_tensor_nd *tensor =
        tc_tensor_nd_from_data(final_shape, final_ndim, data, requires_grad);
    PyMem_Free(data);
    return (PyObject *)tensor_wrap(tensor);
}

static PyObject *tensor_zeros(PyTypeObject *type, PyObject *args,
                              PyObject *kwargs) {
    static char *names[] = {"shape", "requires_grad", NULL};
    PyObject *shape_obj;
    int requires_grad = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|p", names, &shape_obj,
                                     &requires_grad))
        return NULL;
    size_t shape[8], ndim;
    if (parse_shape(shape_obj, shape, &ndim) < 0) return NULL;
    return (PyObject *)tensor_wrap(tc_tensor_nd_create(shape, ndim,
                                                       requires_grad));
}

static PyObject *shape_as_tuple(const tc_tensor_nd *tensor) {
    const size_t ndim = tc_tensor_nd_ndim(tensor);
    const size_t *shape = tc_tensor_nd_shape(tensor);
    PyObject *tuple = PyTuple_New((Py_ssize_t)ndim);
    for (size_t i = 0; i < ndim; ++i)
        PyTuple_SET_ITEM(tuple, (Py_ssize_t)i, PyLong_FromSize_t(shape[i]));
    return tuple;
}

static PyObject *float_list(const float *values, size_t count) {
    PyObject *list = PyList_New((Py_ssize_t)count);
    for (size_t i = 0; i < count; ++i)
        PyList_SET_ITEM(list, (Py_ssize_t)i, PyFloat_FromDouble(values[i]));
    return list;
}

static PyObject *tensor_get_shape(TensorObject *self, void *closure) {
    (void)closure;
    return shape_as_tuple(self->tensor);
}

static PyObject *tensor_get_data(TensorObject *self, void *closure) {
    (void)closure;
    return float_list(tc_tensor_nd_const_data(self->tensor),
                      tc_tensor_nd_numel(self->tensor));
}

static PyObject *tensor_get_grad(TensorObject *self, void *closure) {
    (void)closure;
    const float *grad = tc_tensor_nd_grad(self->tensor);
    if (!grad) Py_RETURN_NONE;
    return float_list(grad, tc_tensor_nd_numel(self->tensor));
}

static PyObject *tensor_backward(TensorObject *self, PyObject *args) {
    (void)args;
    if (!tc_tensor_nd_backward(self->tensor)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "backward requiere un escalar con requires_grad");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *tensor_sgd_step(TensorObject *self, PyObject *args) {
    float learning_rate;
    if (!PyArg_ParseTuple(args, "f", &learning_rate)) return NULL;
    tc_tensor_nd_sgd_step(self->tensor, learning_rate);
    Py_RETURN_NONE;
}

#define BINARY_OP(name, c_name)                                              \
    static PyObject *tensor_##name(TensorObject *self, PyObject *args) {     \
        PyObject *other;                                                     \
        if (!PyArg_ParseTuple(args, "O", &other)) return NULL;               \
        tc_tensor_nd *b = as_tensor(other);                                  \
        if (!b) return NULL;                                                 \
        return (PyObject *)tensor_wrap(c_name(self->tensor, b));             \
    }

BINARY_OP(add, tc_tensor_nd_add)
BINARY_OP(mul, tc_tensor_nd_mul)
BINARY_OP(matmul, tc_tensor_nd_matmul)
BINARY_OP(mse_loss, tc_tensor_nd_mse_loss)
BINARY_OP(cross_entropy, tc_tensor_nd_cross_entropy)
BINARY_OP(embedding, tc_tensor_nd_embedding)
BINARY_OP(cross_entropy_sparse, tc_tensor_nd_cross_entropy_sparse)

static PyObject *tensor_relu(TensorObject *self, PyObject *args) {
    (void)args;
    return (PyObject *)tensor_wrap(tc_tensor_nd_relu(self->tensor));
}

static PyObject *tensor_sigmoid(TensorObject *self, PyObject *args) {
    (void)args;
    return (PyObject *)tensor_wrap(tc_tensor_nd_sigmoid(self->tensor));
}

static PyObject *tensor_tanh(TensorObject *self, PyObject *args) {
    (void)args;
    return (PyObject *)tensor_wrap(tc_tensor_nd_tanh(self->tensor));
}

static PyObject *tensor_softmax(TensorObject *self, PyObject *args) {
    (void)args;
    return (PyObject *)tensor_wrap(tc_tensor_nd_softmax(self->tensor));
}

static PyObject *tensor_sum(TensorObject *self, PyObject *args) {
    (void)args;
    return (PyObject *)tensor_wrap(tc_tensor_nd_sum(self->tensor));
}

static PyObject *tensor_maxpool2d(TensorObject *self, PyObject *args) {
    size_t pool_h, pool_w;
    if (!PyArg_ParseTuple(args, "nn", &pool_h, &pool_w)) return NULL;
    return (PyObject *)tensor_wrap(
        tc_tensor_nd_maxpool2d(self->tensor, pool_h, pool_w));
}

static PyObject *tensor_avgpool2d(TensorObject *self, PyObject *args) {
    size_t pool_h, pool_w;
    if (!PyArg_ParseTuple(args, "nn", &pool_h, &pool_w)) return NULL;
    return (PyObject *)tensor_wrap(
        tc_tensor_nd_avgpool2d(self->tensor, pool_h, pool_w));
}

static PyObject *tensor_save(TensorObject *self, PyObject *args) {
    const char *path;
    if (!PyArg_ParseTuple(args, "s", &path)) return NULL;
    if (!tc_tensor_nd_save(self->tensor, path)) {
        PyErr_SetString(PyExc_IOError, "no se pudo guardar el tensor");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *tensor_copy_(TensorObject *self, PyObject *args) {
    PyObject *src_obj;
    if (!PyArg_ParseTuple(args, "O", &src_obj)) return NULL;
    tc_tensor_nd *src = as_tensor(src_obj);
    if (!src) return NULL;
    if (!tc_tensor_nd_copy_(self->tensor, src)) {
        PyErr_SetString(PyExc_ValueError,
                        "copy_ requiere tensores con la misma forma");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *tensor_reshape(TensorObject *self, PyObject *args) {
    PyObject *shape_obj;
    if (!PyArg_ParseTuple(args, "O", &shape_obj)) return NULL;
    size_t shape[8], ndim;
    if (parse_shape(shape_obj, shape, &ndim) < 0) return NULL;
    return (PyObject *)tensor_wrap(
        tc_tensor_nd_reshape_to(self->tensor, shape, ndim));
}

static PyObject *tensor_conv2d(TensorObject *self, PyObject *args,
                               PyObject *kwargs) {
    static char *names[] = {"kernel", "stride", "padding", NULL};
    PyObject *kernel_obj, *stride_obj = NULL, *padding_obj = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OO", names, &kernel_obj,
                                     &stride_obj, &padding_obj))
        return NULL;
    tc_tensor_nd *kernel = as_tensor(kernel_obj);
    if (!kernel) return NULL;
    tc_conv2d_options options = TC_CONV2D_DEFAULT;
    if (stride_obj) {
        if (!PyArg_ParseTuple(stride_obj, "nn", &options.stride_h,
                              &options.stride_w))
            return NULL;
    }
    if (padding_obj) {
        if (!PyArg_ParseTuple(padding_obj, "nn", &options.pad_h,
                              &options.pad_w))
            return NULL;
    }
    return (PyObject *)tensor_wrap(
        tc_tensor_nd_conv2d_ex(self->tensor, kernel, options));
}

static PyObject *tensor_slice_axis(TensorObject *self, PyObject *args) {
    size_t axis, start, length;
    if (!PyArg_ParseTuple(args, "nnn", &axis, &start, &length)) return NULL;
    return (PyObject *)tensor_wrap(
        tc_tensor_nd_slice_axis(self->tensor, axis, start, length));
}

static PyObject *tensor_transpose(TensorObject *self, PyObject *args) {
    (void)args;
    return (PyObject *)tensor_wrap(tc_tensor_nd_transpose(self->tensor));
}

static PyObject *tensor_layer_norm(TensorObject *self, PyObject *args) {
    (void)args;
    return (PyObject *)tensor_wrap(tc_tensor_nd_layer_norm(self->tensor));
}

/* concat(others, axis=0): others es un Tensor o una secuencia de Tensor.
   self se concatena primero, seguido de los elementos de others. */
static PyObject *tensor_concat(TensorObject *self, PyObject *args,
                               PyObject *kwargs) {
    static char *names[] = {"others", "axis", NULL};
    PyObject *others_obj;
    size_t axis = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|n", names, &others_obj,
                                     &axis))
        return NULL;

    int is_single = PyObject_TypeCheck(others_obj, &TensorType);
    PyObject *seq = is_single ? NULL
        : PySequence_Fast(others_obj,
                          "others debe ser un Tensor o una secuencia");
    const Py_ssize_t extra = is_single ? 1
                                       : PySequence_Fast_GET_SIZE(seq);
    if (!is_single && extra < 0) {
        Py_DECREF(seq);
        return NULL;
    }
    tc_tensor_nd **items = (tc_tensor_nd **)PyMem_Malloc(
        (size_t)(extra + 1) * sizeof(*items));
    if (!items) {
        Py_XDECREF(seq);
        return PyErr_NoMemory();
    }
    items[0] = self->tensor;
    if (is_single) {
        items[1] = as_tensor(others_obj);
        if (!items[1]) {
            PyMem_Free(items);
            return NULL;
        }
    } else {
        for (Py_ssize_t i = 0; i < extra; ++i) {
            items[i + 1] = as_tensor(PySequence_Fast_GET_ITEM(seq, i));
            if (!items[i + 1]) {
                PyMem_Free(items);
                Py_DECREF(seq);
                return NULL;
            }
        }
    }
    tc_tensor_nd *result =
        tc_tensor_nd_concat((const tc_tensor_nd *const *)items,
                            (size_t)(extra + 1), axis);
    PyMem_Free(items);
    Py_XDECREF(seq);
    return (PyObject *)tensor_wrap(result);
}

static PyObject *tensor_attention(TensorObject *self, PyObject *args,
                                  PyObject *kwargs) {
    static char *names[] = {"key", "value", "causal", NULL};
    PyObject *key_obj, *value_obj;
    int causal = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|p", names, &key_obj,
                                     &value_obj, &causal))
        return NULL;
    tc_tensor_nd *key = as_tensor(key_obj);
    if (!key) return NULL;
    tc_tensor_nd *value = as_tensor(value_obj);
    if (!value) return NULL;
    return (PyObject *)tensor_wrap(
        tc_tensor_nd_attention(self->tensor, key, value, causal));
}

static PyObject *tensor_permute(TensorObject *self, PyObject *args) {
    PyObject *axes_obj;
    if (!PyArg_ParseTuple(args, "O", &axes_obj)) return NULL;
    PyObject *seq = PySequence_Fast(axes_obj, "axes debe ser una secuencia");
    if (!seq) return NULL;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    size_t axes[8];
    if (n < 1 || n > 8) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "axes debe tener entre 1 y 8 ejes");
        return NULL;
    }
    for (Py_ssize_t i = 0; i < n; ++i) {
        const long a = PyLong_AsLong(PySequence_Fast_GET_ITEM(seq, i));
        if (a < 0 || PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_SetString(PyExc_ValueError, "ejes inválidos en axes");
            return NULL;
        }
        axes[i] = (size_t)a;
    }
    Py_DECREF(seq);
    return (PyObject *)tensor_wrap(
        tc_tensor_nd_permute(self->tensor, axes, (size_t)n));
}

static PyObject *tensor_multihead_attention(TensorObject *self, PyObject *args,
                                            PyObject *kwargs) {
    static char *names[] = {"key", "value", "heads", "causal", NULL};
    PyObject *key_obj, *value_obj;
    unsigned long heads;
    int causal = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOk|p", names, &key_obj,
                                     &value_obj, &heads, &causal))
        return NULL;
    tc_tensor_nd *key = as_tensor(key_obj);
    if (!key) return NULL;
    tc_tensor_nd *value = as_tensor(value_obj);
    if (!value) return NULL;
    return (PyObject *)tensor_wrap(tc_tensor_nd_multihead_attention(
        self->tensor, key, value, (size_t)heads, causal));
}

static PyMethodDef tensor_methods[] = {
    {"zeros", (PyCFunction)tensor_zeros, METH_VARARGS | METH_KEYWORDS | METH_CLASS,
     "Tensor.zeros(shape, requires_grad=False)"},
    {"backward", (PyCFunction)tensor_backward, METH_NOARGS,
     "Propagates gradients from a scalar loss."},
    {"sgd_step", (PyCFunction)tensor_sgd_step, METH_VARARGS,
     "Applies one SGD update using the current gradient."},
    {"add", (PyCFunction)tensor_add, METH_VARARGS, "Elementwise add."},
    {"mul", (PyCFunction)tensor_mul, METH_VARARGS, "Elementwise multiply."},
    {"matmul", (PyCFunction)tensor_matmul, METH_VARARGS, "Matrix multiply."},
    {"relu", (PyCFunction)tensor_relu, METH_NOARGS, "ReLU."},
    {"sigmoid", (PyCFunction)tensor_sigmoid, METH_NOARGS,
     "Sigmoid (numerically stable)."},
    {"tanh", (PyCFunction)tensor_tanh, METH_NOARGS, "Hyperbolic tangent."},
    {"softmax", (PyCFunction)tensor_softmax, METH_NOARGS,
     "Softmax over the last dimension."},
    {"sum", (PyCFunction)tensor_sum, METH_NOARGS, "Sum of all elements."},
    {"mse_loss", (PyCFunction)tensor_mse_loss, METH_VARARGS, "MSE loss."},
    {"cross_entropy", (PyCFunction)tensor_cross_entropy, METH_VARARGS,
     "Mean cross-entropy over rows."},
    {"embedding", (PyCFunction)tensor_embedding, METH_VARARGS,
     "embedding(ids): gather de filas de la tabla [vocab, dim] según ids."},
    {"cross_entropy_sparse", (PyCFunction)tensor_cross_entropy_sparse,
     METH_VARARGS,
     "cross_entropy_sparse(ids): CE con clase entera por fila [rows]."},
    {"maxpool2d", (PyCFunction)tensor_maxpool2d, METH_VARARGS,
     "maxpool2d(pool_h, pool_w)"},
    {"avgpool2d", (PyCFunction)tensor_avgpool2d, METH_VARARGS,
     "avgpool2d(pool_h, pool_w)"},
    {"save", (PyCFunction)tensor_save, METH_VARARGS,
     "save(path) -> guarda el tensor en un archivo binario."},
    {"copy_", (PyCFunction)tensor_copy_, METH_VARARGS,
     "copy_(src) -> copia los datos de src (misma forma); no toca gradientes."},
    {"reshape", (PyCFunction)tensor_reshape, METH_VARARGS, "reshape(shape)"},
    {"conv2d", (PyCFunction)tensor_conv2d, METH_VARARGS | METH_KEYWORDS,
     "conv2d(kernel, stride=(1,1), padding=(0,0))"},
    {"slice_axis", (PyCFunction)tensor_slice_axis, METH_VARARGS,
     "slice_axis(axis, start, length)"},
    {"transpose", (PyCFunction)tensor_transpose, METH_NOARGS,
     "Transposición de matriz [M, N] -> [N, M]."},
    {"layer_norm", (PyCFunction)tensor_layer_norm, METH_NOARGS,
     "LayerNorm sobre la última dimensión (sin afín)."},
    {"concat", (PyCFunction)tensor_concat, METH_VARARGS | METH_KEYWORDS,
     "concat(others, axis=0): others es un Tensor o secuencia de Tensor."},
    {"attention", (PyCFunction)tensor_attention,
     METH_VARARGS | METH_KEYWORDS,
     "attention(key, value, causal=False) sobre [batch, seq, depth]."},
    {"permute", (PyCFunction)tensor_permute, METH_VARARGS,
     "permute(axes): out[i] = in[axes[i]] por dimensión."},
    {"multihead_attention", (PyCFunction)tensor_multihead_attention,
     METH_VARARGS | METH_KEYWORDS,
     "multihead_attention(key, value, heads, causal=False) sobre [batch, seq, depth]."},
    {NULL, NULL, 0, NULL}
};

static PyGetSetDef tensor_getset[] = {
    {"shape", (getter)tensor_get_shape, NULL, "Tensor shape.", NULL},
    {"data", (getter)tensor_get_data, NULL, "Flat row-major data.", NULL},
    {"grad", (getter)tensor_get_grad, NULL, "Flat gradient or None.", NULL},
    {NULL, NULL, NULL, NULL, NULL}
};

static PyTypeObject TensorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "tensorc.Tensor",
    .tp_basicsize = sizeof(TensorObject),
    .tp_dealloc = (destructor)tensor_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = tensor_methods,
    .tp_getset = tensor_getset,
    .tp_new = tensor_new,
    .tp_doc = "TensorC tensor with automatic differentiation.",
};

/* --- Capa convolucional --- */
typedef struct {
    PyObject_HEAD
    tc_conv2d_layer *layer;
} LayerObject;

static PyTypeObject LayerType;

static void layer_dealloc(LayerObject *self) {
    tc_conv2d_layer_free(self->layer);
    PyObject_Del(self);
}

static PyObject *layer_kernel(LayerObject *self, void *closure) {
    (void)closure;
    tc_tensor_nd_retain(tc_conv2d_layer_kernel(self->layer));
    return (PyObject *)tensor_wrap(tc_conv2d_layer_kernel(self->layer));
}

static PyObject *layer_bias(LayerObject *self, void *closure) {
    (void)closure;
    tc_tensor_nd_retain(tc_conv2d_layer_bias(self->layer));
    return (PyObject *)tensor_wrap(tc_conv2d_layer_bias(self->layer));
}

static PyObject *layer_forward(LayerObject *self, PyObject *args) {
    PyObject *input_obj;
    if (!PyArg_ParseTuple(args, "O", &input_obj)) return NULL;
    tc_tensor_nd *input = as_tensor(input_obj);
    if (!input) return NULL;
    return (PyObject *)tensor_wrap(
        tc_conv2d_layer_forward(self->layer, input));
}

static PyObject *layer_new(PyTypeObject *type, PyObject *args,
                           PyObject *kwargs) {
    static char *names[] = {"in_channels", "out_channels", "kernel_h",
                            "kernel_w", "stride", "padding", "seed", NULL};
    unsigned long in_c, out_c, kh, kw, seed = 1;
    PyObject *stride_obj = NULL, *padding_obj = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "kkkk|OOk", names, &in_c,
                                     &out_c, &kh, &kw, &stride_obj,
                                     &padding_obj, &seed))
        return NULL;
    tc_conv2d_options options = TC_CONV2D_DEFAULT;
    if (stride_obj &&
        !PyArg_ParseTuple(stride_obj, "nn", &options.stride_h,
                          &options.stride_w))
        return NULL;
    if (padding_obj &&
        !PyArg_ParseTuple(padding_obj, "nn", &options.pad_h, &options.pad_w))
        return NULL;
    LayerObject *obj = PyObject_New(LayerObject, &LayerType);
    if (!obj) return NULL;
    obj->layer = tc_conv2d_layer_create(in_c, out_c, kh, kw, options,
                                        (unsigned int)seed);
    if (!obj->layer) {
        PyObject_Del(obj);
        PyErr_SetString(PyExc_RuntimeError,
                        "no se pudo crear la capa convolucional");
        return NULL;
    }
    return (PyObject *)obj;
}

static PyMethodDef layer_methods[] = {
    {"forward", (PyCFunction)layer_forward, METH_VARARGS,
     "forward(input) -> conv + bias"},
    {NULL, NULL, 0, NULL}
};

static PyGetSetDef layer_getset[] = {
    {"kernel", (getter)layer_kernel, NULL, "Trainable kernel tensor.", NULL},
    {"bias", (getter)layer_bias, NULL, "Trainable bias tensor.", NULL},
    {NULL, NULL, NULL, NULL, NULL}
};

static PyTypeObject LayerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "tensorc.Conv2DLayer",
    .tp_basicsize = sizeof(LayerObject),
    .tp_dealloc = (destructor)layer_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = layer_methods,
    .tp_getset = layer_getset,
    .tp_new = layer_new,
    .tp_doc = "Trainable 2D convolution layer.",
};

/* Guarda un Tensor o una secuencia de Tensors como checkpoint. */
static PyObject *module_save(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *obj;
    const char *path;
    if (!PyArg_ParseTuple(args, "Os", &obj, &path)) return NULL;

    PyObject *seq = NULL;
    if (PyObject_TypeCheck(obj, &TensorType)) {
        const tc_tensor_nd *single[] = {((TensorObject *)obj)->tensor};
        if (!tc_save_checkpoint(single, 1, path)) {
            PyErr_SetString(PyExc_IOError, "no se pudo guardar el checkpoint");
            return NULL;
        }
        Py_RETURN_NONE;
    }
    seq = PySequence_Fast(obj, "se esperaba un Tensor o una secuencia de Tensors");
    if (!seq) return NULL;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    const tc_tensor_nd **tensors =
        (const tc_tensor_nd **)malloc((size_t)(n > 0 ? n : 1) * sizeof(*tensors));
    if (!tensors) {
        Py_DECREF(seq);
        return PyErr_NoMemory();
    }
    int valid = 1;
    for (Py_ssize_t i = 0; i < n; ++i) {
        tensors[i] = as_tensor(PySequence_Fast_GET_ITEM(seq, i));
        if (!tensors[i]) {
            valid = 0;
            break;
        }
    }
    const int saved =
        valid && tc_save_checkpoint(tensors, (size_t)n, path);
    free(tensors);
    Py_DECREF(seq);
    if (!saved) {
        if (!PyErr_Occurred())
            PyErr_SetString(PyExc_IOError, "no se pudo guardar el checkpoint");
        return NULL;
    }
    Py_RETURN_NONE;
}

/* Carga un tensor único o un checkpoint; devuelve Tensor o lista de Tensors. */
static PyObject *module_load(PyObject *self, PyObject *args,
                             PyObject *kwargs) {
    (void)self;
    static char *names[] = {"path", "requires_grad", NULL};
    const char *path;
    int requires_grad = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|p", names, &path,
                                     &requires_grad))
        return NULL;

    char magic[8];
    FILE *file = fopen(path, "rb");
    if (!file) {
        PyErr_SetString(PyExc_FileNotFoundError, path);
        return NULL;
    }
    const size_t got = fread(magic, 1, sizeof(magic), file);
    fclose(file);

    if (got == sizeof(magic) && memcmp(magic, "TCND0001", 8) == 0)
        return (PyObject *)tensor_wrap(tc_tensor_nd_load(path, requires_grad));
    if (got == sizeof(magic) && memcmp(magic, "TCKP0001", 8) == 0) {
        size_t count = 0;
        tc_tensor_nd **loaded = tc_load_checkpoint(path, &count, requires_grad);
        if (!loaded) {
            PyErr_SetString(PyExc_RuntimeError, "archivo de checkpoint inválido");
            return NULL;
        }
        PyObject *list = PyList_New((Py_ssize_t)count);
        if (!list) {
            tc_free_checkpoint(loaded, count);
            return NULL;
        }
        for (size_t i = 0; i < count; ++i) {
            TensorObject *item = tensor_wrap(loaded[i]);
            if (!item) {
                /* tensor_wrap ya liberó loaded[i] ante fallo. */
                for (size_t j = i + 1; j < count; ++j)
                    tc_tensor_nd_free(loaded[j]);
                free(loaded);
                Py_DECREF(list);
                return NULL;
            }
            PyList_SET_ITEM(list, (Py_ssize_t)i, (PyObject *)item);
        }
        free(loaded);
        return list;
    }
    PyErr_SetString(PyExc_RuntimeError, "archivo no reconocible como tensorc");
    return NULL;
}

/* Definida más abajo; declaración anticipada para la tabla de métodos. */
static PyObject *module_sinusoidal_positions(PyObject *self, PyObject *args);

/* Backend CUDA: consulta sin #ifdef; sin el backend devuelven False/-1. */
static PyObject *module_cuda_available(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;
    return PyBool_FromLong(tc_cuda_available());
}

static PyObject *module_cuda_set_device(PyObject *self, PyObject *args) {
    int device;
    (void)self;
    if (!PyArg_ParseTuple(args, "i", &device)) return NULL;
    return PyBool_FromLong(tc_cuda_set_device(device));
}

static PyObject *module_cuda_device(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;
    return PyLong_FromLong(tc_cuda_device());
}

static PyMethodDef module_methods[] = {
    {"sinusoidal_positions", (PyCFunction)module_sinusoidal_positions,
     METH_VARARGS,
     "sinusoidal_positions(seq, depth) -> codificación posicional [seq, depth]."},
    {"save", (PyCFunction)module_save, METH_VARARGS,
     "save(tensor_o_lista, path) -> guarda un checkpoint binario."},
    {"load", (PyCFunction)module_load, METH_VARARGS | METH_KEYWORDS,
     "load(path, requires_grad=False) -> Tensor o lista de Tensors."},
    {"cuda_available", (PyCFunction)module_cuda_available, METH_NOARGS,
     "cuda_available() -> True si el paquete se compiló con backend CUDA y hay GPU."},
    {"cuda_set_device", (PyCFunction)module_cuda_set_device, METH_VARARGS,
     "cuda_set_device(i) -> True si el dispositivo i quedó activo."},
    {"cuda_device", (PyCFunction)module_cuda_device, METH_NOARGS,
     "cuda_device() -> índice del dispositivo activo o -1 sin backend."},
    {NULL, NULL, 0, NULL}
};

/* --- Capa densa --- */
typedef struct {
    PyObject_HEAD
    tc_dense_layer *layer;
} DenseLayerObject;

static PyTypeObject DenseLayerType;

static void dense_layer_dealloc(DenseLayerObject *self) {
    tc_dense_layer_free(self->layer);
    PyObject_Del(self);
}

static PyObject *dense_layer_weight(DenseLayerObject *self, void *closure) {
    (void)closure;
    tc_tensor_nd_retain(tc_dense_layer_weight(self->layer));
    return (PyObject *)tensor_wrap(tc_dense_layer_weight(self->layer));
}

static PyObject *dense_layer_bias(DenseLayerObject *self, void *closure) {
    (void)closure;
    tc_tensor_nd_retain(tc_dense_layer_bias(self->layer));
    return (PyObject *)tensor_wrap(tc_dense_layer_bias(self->layer));
}

static PyObject *dense_layer_forward(DenseLayerObject *self, PyObject *args) {
    PyObject *input_obj;
    if (!PyArg_ParseTuple(args, "O", &input_obj)) return NULL;
    tc_tensor_nd *input = as_tensor(input_obj);
    if (!input) return NULL;
    return (PyObject *)tensor_wrap(
        tc_dense_layer_forward(self->layer, input));
}

static PyObject *dense_layer_new(PyTypeObject *type, PyObject *args,
                                 PyObject *kwargs) {
    static char *names[] = {"in_features", "out_features", "seed", NULL};
    unsigned long in_features, out_features, seed = 1;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "kk|k", names, &in_features,
                                     &out_features, &seed))
        return NULL;
    DenseLayerObject *obj = PyObject_New(DenseLayerObject, &DenseLayerType);
    if (!obj) return NULL;
    obj->layer = tc_dense_layer_create(in_features, out_features,
                                       (unsigned int)seed);
    if (!obj->layer) {
        PyObject_Del(obj);
        PyErr_SetString(PyExc_RuntimeError,
                        "no se pudo crear la capa densa");
        return NULL;
    }
    return (PyObject *)obj;
}

static PyMethodDef dense_layer_methods[] = {
    {"forward", (PyCFunction)dense_layer_forward, METH_VARARGS,
     "forward(input) -> matmul + bias"},
    {NULL, NULL, 0, NULL}
};

static PyGetSetDef dense_layer_getset[] = {
    {"weight", (getter)dense_layer_weight, NULL, "Trainable weight tensor.",
     NULL},
    {"bias", (getter)dense_layer_bias, NULL, "Trainable bias tensor.", NULL},
    {NULL, NULL, NULL, NULL, NULL}
};

static PyTypeObject DenseLayerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "tensorc.DenseLayer",
    .tp_basicsize = sizeof(DenseLayerObject),
    .tp_dealloc = (destructor)dense_layer_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = dense_layer_methods,
    .tp_getset = dense_layer_getset,
    .tp_new = dense_layer_new,
    .tp_doc = "Trainable fully-connected layer.",
};

/* --- Capa LSTM --- */
typedef struct {
    PyObject_HEAD
    tc_lstm_layer *layer;
} LSTMLayerObject;

static PyTypeObject LSTMLayerType;

static void lstm_layer_dealloc(LSTMLayerObject *self) {
    tc_lstm_layer_free(self->layer);
    PyObject_Del(self);
}

static PyObject *lstm_layer_weight_ih(LSTMLayerObject *self, void *closure) {
    (void)closure;
    tc_tensor_nd_retain(tc_lstm_layer_weight_ih(self->layer));
    return (PyObject *)tensor_wrap(tc_lstm_layer_weight_ih(self->layer));
}

static PyObject *lstm_layer_weight_hh(LSTMLayerObject *self, void *closure) {
    (void)closure;
    tc_tensor_nd_retain(tc_lstm_layer_weight_hh(self->layer));
    return (PyObject *)tensor_wrap(tc_lstm_layer_weight_hh(self->layer));
}

static PyObject *lstm_layer_bias(LSTMLayerObject *self, void *closure) {
    (void)closure;
    tc_tensor_nd_retain(tc_lstm_layer_bias(self->layer));
    return (PyObject *)tensor_wrap(tc_lstm_layer_bias(self->layer));
}

/* forward(x): x es [seq, batch, in] (time-major); devuelve [seq, batch, hidden]. */
static PyObject *lstm_layer_forward(LSTMLayerObject *self, PyObject *args) {
    PyObject *input_obj;
    if (!PyArg_ParseTuple(args, "O", &input_obj)) return NULL;
    tc_tensor_nd *input = as_tensor(input_obj);
    if (!input) return NULL;
    return (PyObject *)tensor_wrap(tc_lstm_layer_forward(self->layer, input));
}

static PyObject *lstm_layer_new(PyTypeObject *type, PyObject *args,
                                PyObject *kwargs) {
    static char *names[] = {"in_features", "hidden", "seed", NULL};
    unsigned long in_features, hidden, seed = 1;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "kk|k", names, &in_features,
                                     &hidden, &seed))
        return NULL;
    LSTMLayerObject *obj = PyObject_New(LSTMLayerObject, &LSTMLayerType);
    if (!obj) return NULL;
    obj->layer = tc_lstm_layer_create(in_features, hidden, (unsigned int)seed);
    if (!obj->layer) {
        PyObject_Del(obj);
        PyErr_SetString(PyExc_RuntimeError, "no se pudo crear la capa LSTM");
        return NULL;
    }
    return (PyObject *)obj;
}

static PyMethodDef lstm_layer_methods[] = {
    {"forward", (PyCFunction)lstm_layer_forward, METH_VARARGS,
     "forward(x) con x [seq, batch, in] -> [seq, batch, hidden]"},
    {NULL, NULL, 0, NULL}
};

static PyGetSetDef lstm_layer_getset[] = {
    {"weight_ih", (getter)lstm_layer_weight_ih, NULL,
     "Pesos entrada->puertas [4*hidden, in].", NULL},
    {"weight_hh", (getter)lstm_layer_weight_hh, NULL,
     "Pesos oculta->puertas [4*hidden, hidden].", NULL},
    {"bias", (getter)lstm_layer_bias, NULL, "Sesgo de puertas [1, 4*hidden].",
     NULL},
    {NULL, NULL, NULL, NULL, NULL}
};

static PyTypeObject LSTMLayerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "tensorc.LSTMLayer",
    .tp_basicsize = sizeof(LSTMLayerObject),
    .tp_dealloc = (destructor)lstm_layer_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = lstm_layer_methods,
    .tp_getset = lstm_layer_getset,
    .tp_new = lstm_layer_new,
    .tp_doc = "LSTM sobre una secuencia completa [seq, batch, in] "
              "(time-major, h0/c0 en cero).",
};

/* --- Capa Transformer (bloque pre-norm) --- */
typedef struct {
    PyObject_HEAD
    tc_transformer_block *block;
} TransformerBlockObject;

static PyTypeObject TransformerBlockType;

static void transformer_block_dealloc(TransformerBlockObject *self) {
    tc_transformer_block_free(self->block);
    PyObject_Del(self);
}

/* forward(x): x es [batch, seq, model_dim]; devuelve la misma forma. */
static PyObject *transformer_block_forward(TransformerBlockObject *self,
                                           PyObject *args) {
    PyObject *input_obj;
    if (!PyArg_ParseTuple(args, "O", &input_obj)) return NULL;
    tc_tensor_nd *input = as_tensor(input_obj);
    if (!input) return NULL;
    return (PyObject *)tensor_wrap(
        tc_transformer_block_forward(self->block, input));
}

/* params: lista de los 8 tensores entrenables (para Adam). */
static PyObject *transformer_block_params(TransformerBlockObject *self,
                                          void *closure) {
    (void)closure;
    const size_t count = tc_transformer_block_param_count(self->block);
    PyObject *list = PyList_New((Py_ssize_t)count);
    if (!list) return NULL;
    for (size_t i = 0; i < count; ++i) {
        tc_tensor_nd_retain(tc_transformer_block_param(self->block, i));
        PyList_SET_ITEM(list, (Py_ssize_t)i,
                        (PyObject *)tensor_wrap(
                            tc_transformer_block_param(self->block, i)));
    }
    return list;
}

static PyObject *transformer_block_new(PyTypeObject *type, PyObject *args,
                                       PyObject *kwargs) {
    static char *names[] = {"model_dim", "ff_dim", "heads", "seed", NULL};
    unsigned long model_dim, ff_dim, heads, seed = 1;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "kkk|k", names, &model_dim,
                                     &ff_dim, &heads, &seed))
        return NULL;
    TransformerBlockObject *obj =
        PyObject_New(TransformerBlockObject, &TransformerBlockType);
    if (!obj) return NULL;
    obj->block = tc_transformer_block_create(model_dim, ff_dim, heads,
                                             (unsigned int)seed);
    if (!obj->block) {
        PyObject_Del(obj);
        PyErr_SetString(PyExc_RuntimeError,
                        "no se pudo crear el bloque transformer "
                        "(model_dim debe ser divisible entre heads)");
        return NULL;
    }
    return (PyObject *)obj;
}

static PyMethodDef transformer_block_methods[] = {
    {"forward", (PyCFunction)transformer_block_forward, METH_VARARGS,
     "forward(x) con x [batch, seq, model_dim] -> misma forma"},
    {NULL, NULL, 0, NULL}
};

static PyGetSetDef transformer_block_getset[] = {
    {"params", (getter)transformer_block_params, NULL,
     "Lista de los 8 tensores entrenables (para Adam).", NULL},
    {NULL, NULL, NULL, NULL, NULL}
};

static PyTypeObject TransformerBlockType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "tensorc.TransformerBlock",
    .tp_basicsize = sizeof(TransformerBlockObject),
    .tp_dealloc = (destructor)transformer_block_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = transformer_block_methods,
    .tp_getset = transformer_block_getset,
    .tp_new = transformer_block_new,
    .tp_doc = "Bloque Transformer pre-norm (MHA + residual + FFN + residual).",
};

/* Codificación posicional sinusoidal [seq, depth]. */
static PyObject *module_sinusoidal_positions(PyObject *self, PyObject *args) {
    (void)self;
    unsigned long seq, depth;
    if (!PyArg_ParseTuple(args, "kk", &seq, &depth)) return NULL;
    return (PyObject *)tensor_wrap(
        tc_tensor_nd_sinusoidal_positions((size_t)seq, (size_t)depth));
}

/* --- Optimizador Adam --- */
typedef struct {
    PyObject_HEAD
    tc_tensor_nd_optimizer *optimizer;
} OptimizerObject;

static PyTypeObject OptimizerType;

static void optimizer_dealloc(OptimizerObject *self) {
    tc_tensor_nd_optimizer_free(self->optimizer);
    PyObject_Del(self);
}

static PyObject *optimizer_step(OptimizerObject *self, PyObject *args) {
    (void)args;
    return PyLong_FromLong(
        tc_tensor_nd_optimizer_step(self->optimizer));
}

static PyObject *optimizer_set_lr(OptimizerObject *self, PyObject *args) {
    float learning_rate;
    if (!PyArg_ParseTuple(args, "f", &learning_rate)) return NULL;
    tc_tensor_nd_optimizer_set_lr(self->optimizer, learning_rate);
    Py_RETURN_NONE;
}

static PyObject *optimizer_new(PyTypeObject *type, PyObject *args,
                               PyObject *kwargs) {
    static char *names[] = {"params", "learning_rate", NULL};
    PyObject *params_obj;
    float learning_rate;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Of", names, &params_obj,
                                     &learning_rate))
        return NULL;
    PyObject *seq = PySequence_Fast(params_obj,
                                    "params debe ser una secuencia de Tensor");
    if (!seq) return NULL;
    const Py_ssize_t count = PySequence_Fast_GET_SIZE(seq);
    tc_tensor_nd **params =
        (tc_tensor_nd **)PyMem_Malloc((size_t)(count > 0 ? count : 1) *
                                      sizeof(*params));
    if (!params) {
        Py_DECREF(seq);
        return PyErr_NoMemory();
    }
    for (Py_ssize_t i = 0; i < count; ++i) {
        params[i] = as_tensor(PySequence_Fast_GET_ITEM(seq, i));
        if (!params[i]) {
            PyMem_Free(params);
            Py_DECREF(seq);
            return NULL;
        }
    }
    Py_DECREF(seq);
    OptimizerObject *obj = PyObject_New(OptimizerObject, &OptimizerType);
    if (!obj) {
        PyMem_Free(params);
        return NULL;
    }
    obj->optimizer = tc_tensor_nd_optimizer_adam(
        params, (size_t)count, learning_rate);
    PyMem_Free(params);
    if (!obj->optimizer) {
        PyObject_Del(obj);
        PyErr_SetString(PyExc_RuntimeError,
                        "no se pudo crear el optimizador Adam");
        return NULL;
    }
    return (PyObject *)obj;
}

static PyMethodDef optimizer_methods[] = {
    {"step", (PyCFunction)optimizer_step, METH_NOARGS,
     "Aplica un paso de Adam; devuelve cuántos parámetros se actualizaron."},
    {"set_lr", (PyCFunction)optimizer_set_lr, METH_VARARGS,
     "set_lr(lr) -> cambia el learning rate (warmup, schedules)."},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject OptimizerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "tensorc.Adam",
    .tp_basicsize = sizeof(OptimizerObject),
    .tp_dealloc = (destructor)optimizer_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = optimizer_methods,
    .tp_new = optimizer_new,
    .tp_doc = "Adam optimizer over a list of parameter tensors.",
};

static struct PyModuleDef tensorc_module = {
    PyModuleDef_HEAD_INIT, "tensorc._tensorc",
    "TensorC: neural network framework in C11 with autograd.", -1,
    module_methods, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__tensorc(void) {
    if (PyType_Ready(&TensorType) < 0) return NULL;
    if (PyType_Ready(&LayerType) < 0) return NULL;
    if (PyType_Ready(&DenseLayerType) < 0) return NULL;
    if (PyType_Ready(&LSTMLayerType) < 0) return NULL;
    if (PyType_Ready(&TransformerBlockType) < 0) return NULL;
    if (PyType_Ready(&OptimizerType) < 0) return NULL;
    PyObject *module = PyModule_Create(&tensorc_module);
    if (!module) return NULL;
    Py_INCREF(&TensorType);
    if (PyModule_AddObject(module, "Tensor", (PyObject *)&TensorType) < 0) {
        Py_DECREF(&TensorType);
        Py_DECREF(module);
        return NULL;
    }
    Py_INCREF(&LayerType);
    if (PyModule_AddObject(module, "Conv2DLayer", (PyObject *)&LayerType) < 0) {
        Py_DECREF(&LayerType);
        Py_DECREF(module);
        return NULL;
    }
    Py_INCREF(&DenseLayerType);
    if (PyModule_AddObject(module, "DenseLayer",
                           (PyObject *)&DenseLayerType) < 0) {
        Py_DECREF(&DenseLayerType);
        Py_DECREF(module);
        return NULL;
    }
    Py_INCREF(&LSTMLayerType);
    if (PyModule_AddObject(module, "LSTMLayer",
                           (PyObject *)&LSTMLayerType) < 0) {
        Py_DECREF(&LSTMLayerType);
        Py_DECREF(module);
        return NULL;
    }
    Py_INCREF(&TransformerBlockType);
    if (PyModule_AddObject(module, "TransformerBlock",
                           (PyObject *)&TransformerBlockType) < 0) {
        Py_DECREF(&TransformerBlockType);
        Py_DECREF(module);
        return NULL;
    }
    Py_INCREF(&OptimizerType);
    if (PyModule_AddObject(module, "Adam", (PyObject *)&OptimizerType) < 0) {
        Py_DECREF(&OptimizerType);
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
