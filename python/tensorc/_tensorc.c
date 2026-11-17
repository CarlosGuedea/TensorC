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
    {"maxpool2d", (PyCFunction)tensor_maxpool2d, METH_VARARGS,
     "maxpool2d(pool_h, pool_w)"},
    {"reshape", (PyCFunction)tensor_reshape, METH_VARARGS, "reshape(shape)"},
    {"conv2d", (PyCFunction)tensor_conv2d, METH_VARARGS | METH_KEYWORDS,
     "conv2d(kernel, stride=(1,1), padding=(0,0))"},
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

static PyMethodDef module_methods[] = {
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
    Py_INCREF(&OptimizerType);
    if (PyModule_AddObject(module, "Adam", (PyObject *)&OptimizerType) < 0) {
        Py_DECREF(&OptimizerType);
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
