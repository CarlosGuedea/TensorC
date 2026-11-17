#include "minitf.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned int rng_state = 1u;

void mt_seed(unsigned int seed) { rng_state = seed ? seed : 1u; }

static unsigned int mt_rand_u32(void) {
    rng_state = 1664525u * rng_state + 1013904223u;
    return rng_state;
}

static float mt_rand_uniform(void) {
    return (float)(mt_rand_u32() / 4294967296.0);
}

MTTensor mt_tensor_create(size_t rows, size_t cols) {
    MTTensor t = {rows, cols, NULL};
    if (rows == 0 || cols == 0 || rows > ((size_t)-1) / cols || rows * cols > ((size_t)-1) / sizeof(float)) {
        t.rows = t.cols = 0;
        return t;
    }
    t.data = (float *)calloc(rows * cols, sizeof(float));
    if (!t.data) t.rows = t.cols = 0;
    return t;
}

void mt_tensor_free(MTTensor *tensor) {
    if (!tensor) return;
    free(tensor->data);
    tensor->data = NULL;
    tensor->rows = tensor->cols = 0;
}

MTTensor mt_tensor_matmul(const MTTensor *a, const MTTensor *b) {
    MTTensor out = {0, 0, NULL};
    if (!a || !b || !a->data || !b->data || a->cols != b->rows) return out;
    out = mt_tensor_create(a->rows, b->cols);
    if (!out.data) return out;
    for (size_t i = 0; i < a->rows; ++i)
        for (size_t k = 0; k < a->cols; ++k) {
            const float value = a->data[i * a->cols + k];
            for (size_t j = 0; j < b->cols; ++j)
                out.data[i * out.cols + j] += value * b->data[k * b->cols + j];
        }
    return out;
}

MTTensor mt_tensor_add(const MTTensor *a, const MTTensor *b) {
    MTTensor out = {0, 0, NULL};
    if (!a || !b || !a->data || !b->data || a->rows != b->rows || a->cols != b->cols) return out;
    out = mt_tensor_create(a->rows, a->cols);
    if (!out.data) return out;
    for (size_t i = 0; i < a->rows * a->cols; ++i) out.data[i] = a->data[i] + b->data[i];
    return out;
}

void mt_tensor_print(const MTTensor *tensor, const char *name) {
    if (!tensor || !tensor->data) { printf("%s: <tensor inválido>\n", name ? name : "tensor"); return; }
    printf("%s [%zu x %zu]\n", name ? name : "tensor", tensor->rows, tensor->cols);
    for (size_t i = 0; i < tensor->rows; ++i) {
        for (size_t j = 0; j < tensor->cols; ++j)
            printf("%s% .5f", j ? "  " : "  ", tensor->data[i * tensor->cols + j]);
        putchar('\n');
    }
}

static float activate(float x, MTActivation activation) {
    switch (activation) {
        case MT_RELU: return x > 0.0f ? x : 0.0f;
        case MT_SIGMOID:
            if (x >= 0.0f) { float z = expf(-x); return 1.0f / (1.0f + z); }
            else { float z = expf(x); return z / (1.0f + z); }
        case MT_TANH: return tanhf(x);
        case MT_LINEAR: default: return x;
    }
}

static float activation_grad(float z, float a, MTActivation activation) {
    switch (activation) {
        case MT_RELU: return z > 0.0f ? 1.0f : 0.0f;
        case MT_SIGMOID: return a * (1.0f - a);
        case MT_TANH: return 1.0f - a * a;
        case MT_LINEAR: default: return 1.0f;
    }
}

MTModel mt_model_create(size_t input_size, const size_t *layer_sizes,
                        const MTActivation *activations, size_t layer_count) {
    MTModel model = {input_size, 0, NULL};
    if (!input_size || !layer_sizes || !activations || !layer_count) return model;
    model.layers = (MTLayer *)calloc(layer_count, sizeof(MTLayer));
    if (!model.layers) return model;
    size_t in = input_size;
    for (size_t l = 0; l < layer_count; ++l) {
        MTLayer *layer = &model.layers[l];
        layer->input_size = in;
        layer->output_size = layer_sizes[l];
        layer->activation = activations[l];
        if (!layer->output_size || in > ((size_t)-1) / layer->output_size ||
            in * layer->output_size > ((size_t)-1) / sizeof(float)) {
            mt_model_free(&model); return model;
        }
        layer->weights = (float *)malloc(in * layer->output_size * sizeof(float));
        layer->biases = (float *)calloc(layer->output_size, sizeof(float));
        if (!layer->weights || !layer->biases) { mt_model_free(&model); return model; }
        /* Xavier uniform initialization. */
        float limit = sqrtf(6.0f / (float)(in + layer->output_size));
        for (size_t i = 0; i < in * layer->output_size; ++i)
            layer->weights[i] = (2.0f * mt_rand_uniform() - 1.0f) * limit;
        in = layer->output_size;
        model.layer_count++;
    }
    return model;
}

void mt_model_free(MTModel *model) {
    if (!model) return;
    for (size_t l = 0; l < model->layer_count; ++l) {
        free(model->layers[l].weights);
        free(model->layers[l].biases);
    }
    free(model->layers);
    model->layers = NULL;
    model->layer_count = model->input_size = 0;
}

static int valid_input(const MTModel *model, const MTTensor *input) {
    return model && model->layers && model->layer_count && input && input->data &&
           input->rows == 1 && input->cols == model->input_size;
}

MTTensor mt_model_predict(const MTModel *model, const MTTensor *input) {
    MTTensor invalid = {0, 0, NULL};
    if (!valid_input(model, input)) return invalid;
    MTTensor current = mt_tensor_create(1, input->cols);
    if (!current.data) return invalid;
    memcpy(current.data, input->data, input->cols * sizeof(float));
    for (size_t l = 0; l < model->layer_count; ++l) {
        const MTLayer *layer = &model->layers[l];
        MTTensor next = mt_tensor_create(1, layer->output_size);
        if (!next.data) { mt_tensor_free(&current); return invalid; }
        for (size_t j = 0; j < layer->output_size; ++j) {
            float z = layer->biases[j];
            for (size_t i = 0; i < layer->input_size; ++i)
                z += current.data[i] * layer->weights[i * layer->output_size + j];
            next.data[j] = activate(z, layer->activation);
        }
        mt_tensor_free(&current);
        current = next;
    }
    return current;
}

float mt_model_train_sample(MTModel *model, const MTTensor *input,
                            const MTTensor *target, float learning_rate) {
    if (!valid_input(model, input) || !target || !target->data || target->rows != 1 ||
        target->cols != model->layers[model->layer_count - 1].output_size || learning_rate <= 0.0f)
        return NAN;
    const size_t L = model->layer_count;
    float **acts = (float **)calloc(L + 1, sizeof(float *));
    float **zs = (float **)calloc(L, sizeof(float *));
    float **deltas = (float **)calloc(L, sizeof(float *));
    if (!acts || !zs || !deltas) { free(acts); free(zs); free(deltas); return NAN; }
    acts[0] = (float *)malloc(input->cols * sizeof(float));
    if (!acts[0]) goto fail;
    memcpy(acts[0], input->data, input->cols * sizeof(float));
    for (size_t l = 0; l < L; ++l) {
        MTLayer *layer = &model->layers[l];
        acts[l + 1] = (float *)malloc(layer->output_size * sizeof(float));
        zs[l] = (float *)malloc(layer->output_size * sizeof(float));
        deltas[l] = (float *)malloc(layer->output_size * sizeof(float));
        if (!acts[l + 1] || !zs[l] || !deltas[l]) goto fail;
        for (size_t j = 0; j < layer->output_size; ++j) {
            float z = layer->biases[j];
            for (size_t i = 0; i < layer->input_size; ++i)
                z += acts[l][i] * layer->weights[i * layer->output_size + j];
            zs[l][j] = z;
            acts[l + 1][j] = activate(z, layer->activation);
        }
    }
    MTLayer *last = &model->layers[L - 1];
    float loss = 0.0f;
    for (size_t j = 0; j < last->output_size; ++j) {
        float error = acts[L][j] - target->data[j];
        loss += error * error;
        deltas[L - 1][j] = error * activation_grad(zs[L - 1][j], acts[L][j], last->activation);
    }
    loss /= (float)last->output_size;
    for (size_t rev = L - 1; rev > 0; --rev) {
        MTLayer *layer = &model->layers[rev - 1];
        MTLayer *next = &model->layers[rev];
        for (size_t i = 0; i < layer->output_size; ++i) {
            float propagated = 0.0f;
            for (size_t j = 0; j < next->output_size; ++j)
                propagated += next->weights[i * next->output_size + j] * deltas[rev][j];
            deltas[rev - 1][i] = propagated * activation_grad(zs[rev - 1][i], acts[rev][i], layer->activation);
        }
    }
    for (size_t l = 0; l < L; ++l) {
        MTLayer *layer = &model->layers[l];
        for (size_t i = 0; i < layer->input_size; ++i)
            for (size_t j = 0; j < layer->output_size; ++j)
                layer->weights[i * layer->output_size + j] -= learning_rate * acts[l][i] * deltas[l][j];
        for (size_t j = 0; j < layer->output_size; ++j)
            layer->biases[j] -= learning_rate * deltas[l][j];
    }
    for (size_t l = 0; l <= L; ++l) free(acts[l]);
    for (size_t l = 0; l < L; ++l) { free(zs[l]); free(deltas[l]); }
    free(acts); free(zs); free(deltas);
    return loss;
fail:
    if (acts) { for (size_t l = 0; l <= L; ++l) free(acts[l]); }
    if (zs) { for (size_t l = 0; l < L; ++l) free(zs[l]); }
    if (deltas) { for (size_t l = 0; l < L; ++l) free(deltas[l]); }
    free(acts); free(zs); free(deltas);
    return NAN;
}

float mt_model_train_epoch(MTModel *model, const MTTensor *x,
                           const MTTensor *y, float learning_rate) {
    if (!model || !x || !y || !x->data || !y->data || x->rows != y->rows ||
        x->cols != model->input_size || x->rows == 0) return NAN;
    float total = 0.0f;
    MTTensor xi = mt_tensor_create(1, x->cols);
    MTTensor yi = mt_tensor_create(1, y->cols);
    if (!xi.data || !yi.data) { mt_tensor_free(&xi); mt_tensor_free(&yi); return NAN; }
    for (size_t r = 0; r < x->rows; ++r) {
        memcpy(xi.data, &x->data[r * x->cols], x->cols * sizeof(float));
        memcpy(yi.data, &y->data[r * y->cols], y->cols * sizeof(float));
        total += mt_model_train_sample(model, &xi, &yi, learning_rate);
    }
    mt_tensor_free(&xi); mt_tensor_free(&yi);
    return total / (float)x->rows;
}
