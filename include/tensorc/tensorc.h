#ifndef TENSORC_TENSORC_H
#define TENSORC_TENSORC_H

/* TensorC 0.1 public API. This release focuses on CPU dense neural networks. */
#include <minitf.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TENSORC_VERSION "0.1.0"

typedef MTTensor tc_tensor;
typedef MTActivation tc_activation;
typedef MTModel tc_model;

#define TC_LINEAR MT_LINEAR
#define TC_RELU MT_RELU
#define TC_SIGMOID MT_SIGMOID
#define TC_TANH MT_TANH

#define tc_seed mt_seed
#define tc_tensor_create mt_tensor_create
#define tc_tensor_free mt_tensor_free
#define tc_tensor_matmul mt_tensor_matmul
#define tc_tensor_add mt_tensor_add
#define tc_tensor_print mt_tensor_print
#define tc_model_create mt_model_create
#define tc_model_free mt_model_free
#define tc_model_predict mt_model_predict
#define tc_model_train_sample mt_model_train_sample
#define tc_model_train_epoch mt_model_train_epoch

#ifdef __cplusplus
}
#endif

#endif
