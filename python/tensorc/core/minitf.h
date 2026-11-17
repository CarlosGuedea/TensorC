#ifndef MINITF_H
#define MINITF_H

#include <stddef.h>

/* MiniTF: tiny, dependency-free C11 neural-network library (2-D row-major tensors). */
typedef struct {
    size_t rows;
    size_t cols;
    float *data;
} MTTensor;

typedef enum {
    MT_LINEAR = 0,
    MT_RELU = 1,
    MT_SIGMOID = 2,
    MT_TANH = 3
} MTActivation;

typedef struct {
    size_t input_size;
    size_t output_size;
    MTActivation activation;
    float *weights; /* [input_size, output_size] */
    float *biases;  /* [output_size] */
} MTLayer;

typedef struct {
    size_t input_size;
    size_t layer_count;
    MTLayer *layers;
} MTModel;

typedef enum {
    MT_OPT_SGD = 0,
    MT_OPT_ADAM = 1
} MTOptimizerType;

typedef struct {
    MTOptimizerType type;
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    size_t step;
    size_t layer_count;
    float **m_weights;
    float **v_weights;
    float **m_biases;
    float **v_biases;
} MTOptimizer;

MTTensor mt_tensor_create(size_t rows, size_t cols);
void mt_tensor_free(MTTensor *tensor);
MTTensor mt_tensor_matmul(const MTTensor *a, const MTTensor *b);
MTTensor mt_tensor_add(const MTTensor *a, const MTTensor *b);
void mt_tensor_print(const MTTensor *tensor, const char *name);

/* Seed the small deterministic PRNG used for weight initialization. */
void mt_seed(unsigned int seed);
MTModel mt_model_create(size_t input_size, const size_t *layer_sizes,
                        const MTActivation *activations, size_t layer_count);
void mt_model_free(MTModel *model);
/* Save/load the dense model parameters using TensorC's versioned binary format. */
int mt_model_save(const MTModel *model, const char *path);
MTModel mt_model_load(const char *path);
MTOptimizer mt_optimizer_sgd(float learning_rate);
MTOptimizer mt_optimizer_adam(const MTModel *model, float learning_rate);
void mt_optimizer_free(MTOptimizer *optimizer);
/* Input and output are single-row tensors. Caller owns the returned tensor. */
MTTensor mt_model_predict(const MTModel *model, const MTTensor *input);
/* One online SGD update; returns mean squared error before the update. */
float mt_model_train_sample(MTModel *model, const MTTensor *input,
                            const MTTensor *target, float learning_rate);
float mt_model_train_sample_opt(MTModel *model, const MTTensor *input,
                                const MTTensor *target, MTOptimizer *optimizer);
/* Train over all samples for one epoch; X and Y have one sample per row. */
float mt_model_train_epoch(MTModel *model, const MTTensor *x,
                           const MTTensor *y, float learning_rate);
/* Optimizer-aware training. batch_size controls when gradients are applied. */
float mt_model_train_batch(MTModel *model, const MTTensor *x,
                           const MTTensor *y, MTOptimizer *optimizer,
                           size_t batch_size);
float mt_model_train_epoch_opt(MTModel *model, const MTTensor *x,
                               const MTTensor *y, MTOptimizer *optimizer,
                               size_t batch_size);

#endif
