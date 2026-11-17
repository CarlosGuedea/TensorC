#ifndef TENSORC_TENSORC_H
#define TENSORC_TENSORC_H

/* TensorC public API: compatible dense models plus N-dimensional autograd. */
#include <minitf.h>
#include <tensorc/ndtensor.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TENSORC_VERSION "0.9.0"

typedef MTTensor tc_tensor;
typedef MTActivation tc_activation;
typedef MTModel tc_model;
typedef MTOptimizer tc_optimizer;

#define TC_LINEAR MT_LINEAR
#define TC_RELU MT_RELU
#define TC_SIGMOID MT_SIGMOID
#define TC_TANH MT_TANH
#define TC_OPT_SGD MT_OPT_SGD
#define TC_OPT_ADAM MT_OPT_ADAM

#define tc_seed mt_seed
#define tc_tensor_create mt_tensor_create
#define tc_tensor_free mt_tensor_free
#define tc_tensor_matmul mt_tensor_matmul
#define tc_tensor_add mt_tensor_add
#define tc_tensor_print mt_tensor_print
#define tc_model_create mt_model_create
#define tc_model_free mt_model_free
#define tc_model_save mt_model_save
#define tc_model_load mt_model_load
#define tc_optimizer_sgd mt_optimizer_sgd
#define tc_optimizer_adam mt_optimizer_adam
#define tc_optimizer_free mt_optimizer_free
#define tc_model_predict mt_model_predict
#define tc_model_train_sample mt_model_train_sample
#define tc_model_train_sample_opt mt_model_train_sample_opt
#define tc_model_train_epoch mt_model_train_epoch
#define tc_model_train_batch mt_model_train_batch
#define tc_model_train_epoch_opt mt_model_train_epoch_opt

#ifdef __cplusplus
}
#endif

#endif
