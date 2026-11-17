#include "minitf.h"

#include <math.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef TENSORC_HAS_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

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
#ifdef TENSORC_HAS_ACCELERATE
    const size_t work = a->rows * a->cols * b->cols;
    if (work >= 4096 && a->rows <= (size_t)INT_MAX && a->cols <= (size_t)INT_MAX &&
        b->cols <= (size_t)INT_MAX) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    (int)a->rows, (int)b->cols, (int)a->cols,
                    1.0f, a->data, (int)a->cols,
                    b->data, (int)b->cols,
                    0.0f, out.data, (int)out.cols);
        return out;
    }
#endif
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

typedef struct {
    size_t layer_count;
    float **weights;
    float **biases;
} MTGradients;

/* Reused forward/backward buffers; avoids heap traffic for every sample. */
typedef struct {
    size_t layer_count;
    float **acts;
    float **zs;
    float **deltas;
} MTWorkspace;

static MTWorkspace workspace_create(const MTModel *model) {
    MTWorkspace workspace = {0, NULL, NULL, NULL};
    if (!model || !model->layers || model->layer_count == 0) return workspace;
    workspace.layer_count = model->layer_count;
    workspace.acts = (float **)calloc(workspace.layer_count + 1, sizeof(float *));
    workspace.zs = (float **)calloc(workspace.layer_count, sizeof(float *));
    workspace.deltas = (float **)calloc(workspace.layer_count, sizeof(float *));
    if (!workspace.acts || !workspace.zs || !workspace.deltas) goto fail;
    for (size_t l = 0; l < workspace.layer_count; ++l) {
        const MTLayer *layer = &model->layers[l];
        workspace.acts[l + 1] = (float *)malloc(layer->output_size * sizeof(float));
        workspace.zs[l] = (float *)malloc(layer->output_size * sizeof(float));
        workspace.deltas[l] = (float *)malloc(layer->output_size * sizeof(float));
        if (!workspace.acts[l + 1] || !workspace.zs[l] || !workspace.deltas[l]) goto fail;
    }
    return workspace;
fail:
    for (size_t l = 0; l < workspace.layer_count; ++l) {
        free(workspace.acts ? workspace.acts[l + 1] : NULL);
        free(workspace.zs ? workspace.zs[l] : NULL);
        free(workspace.deltas ? workspace.deltas[l] : NULL);
    }
    free(workspace.acts);
    free(workspace.zs);
    free(workspace.deltas);
    return (MTWorkspace){0, NULL, NULL, NULL};
}

static void workspace_free(MTWorkspace *workspace) {
    if (!workspace) return;
    for (size_t l = 0; l < workspace->layer_count; ++l) {
        free(workspace->acts[l + 1]);
        free(workspace->zs[l]);
        free(workspace->deltas[l]);
    }
    free(workspace->acts);
    free(workspace->zs);
    free(workspace->deltas);
    *workspace = (MTWorkspace){0, NULL, NULL, NULL};
}

static MTGradients gradients_create(const MTModel *model) {
    MTGradients gradients = {0, NULL, NULL};
    if (!model || !model->layers || model->layer_count == 0) return gradients;
    gradients.layer_count = model->layer_count;
    gradients.weights = (float **)calloc(gradients.layer_count, sizeof(float *));
    gradients.biases = (float **)calloc(gradients.layer_count, sizeof(float *));
    if (!gradients.weights || !gradients.biases) goto fail;
    for (size_t l = 0; l < gradients.layer_count; ++l) {
        const MTLayer *layer = &model->layers[l];
        gradients.weights[l] = (float *)calloc(layer->input_size * layer->output_size, sizeof(float));
        gradients.biases[l] = (float *)calloc(layer->output_size, sizeof(float));
        if (!gradients.weights[l] || !gradients.biases[l]) goto fail;
    }
    return gradients;
fail:
    for (size_t l = 0; l < gradients.layer_count; ++l) {
        free(gradients.weights ? gradients.weights[l] : NULL);
        free(gradients.biases ? gradients.biases[l] : NULL);
    }
    free(gradients.weights);
    free(gradients.biases);
    gradients.layer_count = 0;
    gradients.weights = gradients.biases = NULL;
    return gradients;
}

static void gradients_free(MTGradients *gradients) {
    if (!gradients) return;
    for (size_t l = 0; l < gradients->layer_count; ++l) {
        free(gradients->weights[l]);
        free(gradients->biases[l]);
    }
    free(gradients->weights);
    free(gradients->biases);
    gradients->layer_count = 0;
    gradients->weights = gradients->biases = NULL;
}

static void gradients_zero(const MTModel *model, MTGradients *gradients) {
    for (size_t l = 0; l < model->layer_count; ++l) {
        const MTLayer *layer = &model->layers[l];
        memset(gradients->weights[l], 0, layer->input_size * layer->output_size * sizeof(float));
        memset(gradients->biases[l], 0, layer->output_size * sizeof(float));
    }
}

static void gradients_add(const MTModel *model, MTGradients *destination,
                          const MTGradients *source) {
    for (size_t l = 0; l < model->layer_count; ++l) {
        const MTLayer *layer = &model->layers[l];
        for (size_t i = 0; i < layer->input_size * layer->output_size; ++i)
            destination->weights[l][i] += source->weights[l][i];
        for (size_t j = 0; j < layer->output_size; ++j)
            destination->biases[l][j] += source->biases[l][j];
    }
}

static float compute_gradients(const MTModel *model, const MTTensor *input,
                               const MTTensor *target, MTGradients *gradients,
                               MTWorkspace *workspace) {
    if (!valid_input(model, input) || !target || !target->data || target->rows != 1 ||
        target->cols != model->layers[model->layer_count - 1].output_size ||
        !gradients || gradients->layer_count != model->layer_count || !workspace ||
        workspace->layer_count != model->layer_count) return NAN;

    const size_t L = model->layer_count;
    float **acts = workspace->acts;
    float **zs = workspace->zs;
    float **deltas = workspace->deltas;
    gradients_zero(model, gradients);
    acts[0] = input->data;

    for (size_t l = 0; l < L; ++l) {
        MTLayer *layer = &model->layers[l];
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
        const float error = acts[L][j] - target->data[j];
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
        const MTLayer *layer = &model->layers[l];
        for (size_t i = 0; i < layer->input_size; ++i)
            for (size_t j = 0; j < layer->output_size; ++j)
                gradients->weights[l][i * layer->output_size + j] = acts[l][i] * deltas[l][j];
        for (size_t j = 0; j < layer->output_size; ++j)
            gradients->biases[l][j] = deltas[l][j];
    }

    return loss;
}

MTOptimizer mt_optimizer_sgd(float learning_rate) {
    MTOptimizer optimizer = {0};
    optimizer.type = MT_OPT_SGD;
    optimizer.learning_rate = learning_rate;
    optimizer.epsilon = 1e-8f;
    return optimizer;
}

MTOptimizer mt_optimizer_adam(const MTModel *model, float learning_rate) {
    MTOptimizer optimizer = {0};
    if (!model || !model->layers || model->layer_count == 0 || learning_rate <= 0.0f) return optimizer;
    optimizer.type = MT_OPT_ADAM;
    optimizer.learning_rate = learning_rate;
    optimizer.beta1 = 0.9f;
    optimizer.beta2 = 0.999f;
    optimizer.epsilon = 1e-8f;
    optimizer.layer_count = model->layer_count;
    optimizer.m_weights = (float **)calloc(optimizer.layer_count, sizeof(float *));
    optimizer.v_weights = (float **)calloc(optimizer.layer_count, sizeof(float *));
    optimizer.m_biases = (float **)calloc(optimizer.layer_count, sizeof(float *));
    optimizer.v_biases = (float **)calloc(optimizer.layer_count, sizeof(float *));
    if (!optimizer.m_weights || !optimizer.v_weights || !optimizer.m_biases || !optimizer.v_biases) goto fail;
    for (size_t l = 0; l < optimizer.layer_count; ++l) {
        const MTLayer *layer = &model->layers[l];
        const size_t weight_count = layer->input_size * layer->output_size;
        optimizer.m_weights[l] = (float *)calloc(weight_count, sizeof(float));
        optimizer.v_weights[l] = (float *)calloc(weight_count, sizeof(float));
        optimizer.m_biases[l] = (float *)calloc(layer->output_size, sizeof(float));
        optimizer.v_biases[l] = (float *)calloc(layer->output_size, sizeof(float));
        if (!optimizer.m_weights[l] || !optimizer.v_weights[l] ||
            !optimizer.m_biases[l] || !optimizer.v_biases[l]) goto fail;
    }
    return optimizer;
fail:
    mt_optimizer_free(&optimizer);
    return (MTOptimizer){0};
}

void mt_optimizer_free(MTOptimizer *optimizer) {
    if (!optimizer) return;
    for (size_t l = 0; l < optimizer->layer_count; ++l) {
        free(optimizer->m_weights ? optimizer->m_weights[l] : NULL);
        free(optimizer->v_weights ? optimizer->v_weights[l] : NULL);
        free(optimizer->m_biases ? optimizer->m_biases[l] : NULL);
        free(optimizer->v_biases ? optimizer->v_biases[l] : NULL);
    }
    free(optimizer->m_weights);
    free(optimizer->v_weights);
    free(optimizer->m_biases);
    free(optimizer->v_biases);
    *optimizer = (MTOptimizer){0};
}

static int optimizer_valid(const MTModel *model, const MTOptimizer *optimizer) {
    if (!model || !optimizer || optimizer->learning_rate <= 0.0f) return 0;
    if (optimizer->type == MT_OPT_SGD) return 1;
    return optimizer->type == MT_OPT_ADAM && optimizer->layer_count == model->layer_count &&
           optimizer->m_weights && optimizer->v_weights && optimizer->m_biases && optimizer->v_biases;
}

static void apply_gradients(MTModel *model, const MTGradients *gradients,
                            MTOptimizer *optimizer, float scale) {
    optimizer->step++;
    const float t = (float)optimizer->step;
    const float correction1 = 1.0f - powf(optimizer->beta1, t);
    const float correction2 = 1.0f - powf(optimizer->beta2, t);
    for (size_t l = 0; l < model->layer_count; ++l) {
        MTLayer *layer = &model->layers[l];
        const size_t weight_count = layer->input_size * layer->output_size;
        for (size_t i = 0; i < weight_count; ++i) {
            const float gradient = gradients->weights[l][i] * scale;
            if (optimizer->type == MT_OPT_ADAM) {
                optimizer->m_weights[l][i] = optimizer->beta1 * optimizer->m_weights[l][i] + (1.0f - optimizer->beta1) * gradient;
                optimizer->v_weights[l][i] = optimizer->beta2 * optimizer->v_weights[l][i] + (1.0f - optimizer->beta2) * gradient * gradient;
                const float m_hat = optimizer->m_weights[l][i] / correction1;
                const float v_hat = optimizer->v_weights[l][i] / correction2;
                layer->weights[i] -= optimizer->learning_rate * m_hat / (sqrtf(v_hat) + optimizer->epsilon);
            } else {
                layer->weights[i] -= optimizer->learning_rate * gradient;
            }
        }
        for (size_t j = 0; j < layer->output_size; ++j) {
            const float gradient = gradients->biases[l][j] * scale;
            if (optimizer->type == MT_OPT_ADAM) {
                optimizer->m_biases[l][j] = optimizer->beta1 * optimizer->m_biases[l][j] + (1.0f - optimizer->beta1) * gradient;
                optimizer->v_biases[l][j] = optimizer->beta2 * optimizer->v_biases[l][j] + (1.0f - optimizer->beta2) * gradient * gradient;
                const float m_hat = optimizer->m_biases[l][j] / correction1;
                const float v_hat = optimizer->v_biases[l][j] / correction2;
                layer->biases[j] -= optimizer->learning_rate * m_hat / (sqrtf(v_hat) + optimizer->epsilon);
            } else {
                layer->biases[j] -= optimizer->learning_rate * gradient;
            }
        }
    }
}

float mt_model_train_sample_opt(MTModel *model, const MTTensor *input,
                                const MTTensor *target, MTOptimizer *optimizer) {
    if (!optimizer_valid(model, optimizer)) return NAN;
    MTGradients gradients = gradients_create(model);
    MTWorkspace workspace = workspace_create(model);
    if (!gradients.weights || !workspace.acts) {
        gradients_free(&gradients);
        workspace_free(&workspace);
        return NAN;
    }
    const float loss = compute_gradients(model, input, target, &gradients, &workspace);
    if (!isnan(loss)) apply_gradients(model, &gradients, optimizer, 1.0f);
    gradients_free(&gradients);
    workspace_free(&workspace);
    return loss;
}

float mt_model_train_sample(MTModel *model, const MTTensor *input,
                            const MTTensor *target, float learning_rate) {
    MTOptimizer optimizer = mt_optimizer_sgd(learning_rate);
    return mt_model_train_sample_opt(model, input, target, &optimizer);
}

float mt_model_train_batch(MTModel *model, const MTTensor *x,
                           const MTTensor *y, MTOptimizer *optimizer,
                           size_t batch_size) {
    if (!optimizer_valid(model, optimizer) || !x || !y || !x->data || !y->data ||
        x->rows != y->rows || x->rows == 0 || x->cols != model->input_size || batch_size == 0 ||
        y->cols != model->layers[model->layer_count - 1].output_size) return NAN;
    if (batch_size > x->rows) batch_size = x->rows;
    MTGradients accumulated = gradients_create(model);
    MTGradients sample = gradients_create(model);
    MTWorkspace workspace = workspace_create(model);
    if (!accumulated.weights || !sample.weights || !workspace.acts) {
        gradients_free(&accumulated); gradients_free(&sample);
        workspace_free(&workspace);
        return NAN;
    }
    float total_loss = 0.0f;
    size_t processed = 0;
    while (processed < x->rows) {
        const size_t count = (x->rows - processed < batch_size) ? x->rows - processed : batch_size;
        gradients_zero(model, &accumulated);
        for (size_t offset = 0; offset < count; ++offset) {
            const size_t row = processed + offset;
            const MTTensor xi = {1, x->cols, &x->data[row * x->cols]};
            const MTTensor yi = {1, y->cols, &y->data[row * y->cols]};
            const float loss = compute_gradients(model, &xi, &yi, &sample, &workspace);
            if (isnan(loss)) {
                gradients_free(&accumulated); gradients_free(&sample);
                workspace_free(&workspace);
                return NAN;
            }
            total_loss += loss;
            gradients_add(model, &accumulated, &sample);
        }
        apply_gradients(model, &accumulated, optimizer, 1.0f / (float)count);
        processed += count;
    }
    gradients_free(&accumulated); gradients_free(&sample);
    workspace_free(&workspace);
    return total_loss / (float)x->rows;
}

float mt_model_train_epoch_opt(MTModel *model, const MTTensor *x,
                               const MTTensor *y, MTOptimizer *optimizer,
                               size_t batch_size) {
    if (!x || x->rows == 0) return NAN;
    if (batch_size == 0) batch_size = x->rows;
    return mt_model_train_batch(model, x, y, optimizer, batch_size);
}

float mt_model_train_epoch(MTModel *model, const MTTensor *x,
                           const MTTensor *y, float learning_rate) {
    MTOptimizer optimizer = mt_optimizer_sgd(learning_rate);
    return mt_model_train_epoch_opt(model, x, y, &optimizer, 1);
}
