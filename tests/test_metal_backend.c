#include <tensorc/ndtensor.h>
#ifdef TENSORC_HAS_METAL
#include "tensorc_device_internal.h"
#endif

#include <math.h>
#include <stdio.h>

static int close_enough(float a, float b) { return fabsf(a - b) < 1e-4f; }

int main(void) {
#ifndef TENSORC_HAS_METAL
    if (tc_metal_available()) return 1;
    puts("Metal unavailable in CPU build: ok");
    return 0;
#else
    if (!tc_metal_available()) {
        fputs("Metal build did not find the Apple GPU\n", stderr);
        return 1;
    }
    float *device_a = tc__device_alloc(6);
    float *device_b = tc__device_alloc(6);
    float *device_c = tc__device_alloc(4);
    const float ma[] = {1, 2, 3, 4, 5, 6};
    const float mb[] = {7, 8, 9, 10, 11, 12};
    float mc[4] = {0};
    int device_ok = device_a && device_b && device_c &&
        tc__device_copy_h2d(device_a, ma, 6) &&
        tc__device_copy_h2d(device_b, mb, 6) &&
        tc__device_matmul_nn(device_a, device_b, device_c, 2, 3, 2) &&
        tc__device_copy_d2h(mc, device_c, 4);
    for (size_t i = 0; device_ok && i < 4; ++i)
        device_ok = close_enough(mc[i], (float[]){58, 64, 139, 154}[i]);
    tc__device_free(device_a, 6);
    tc__device_free(device_b, 6);
    tc__device_free(device_c, 4);
    if (!device_ok) return 1;
    const size_t x_shape[] = {2, 2}, bias_shape[] = {1, 2};
    const float xv[] = {-1, 2, 3, -4}, biasv[] = {2, -1};
    const float expected_relu[] = {1, 1, 5, 0};
    tc_tensor_nd *x = tc_tensor_nd_from_data(x_shape, 2, xv, 1);
    tc_tensor_nd *bias = tc_tensor_nd_from_data(bias_shape, 2, biasv, 1);
    tc_tensor_nd *sum0 = tc_tensor_nd_add(x, bias);
    tc_tensor_nd *relu = tc_tensor_nd_relu(sum0);
    tc_tensor_nd *loss0 = tc_tensor_nd_sum(relu);
    int graph_ok = x && bias && sum0 && relu && loss0 &&
                   tc_tensor_nd_backward(loss0);
    const float *relu_values = tc_tensor_nd_const_data(relu);
    const float *x_grad = tc_tensor_nd_grad(x);
    const float *bias_grad = tc_tensor_nd_grad(bias);
    for (size_t i = 0; graph_ok && i < 4; ++i)
        graph_ok = close_enough(relu_values[i], expected_relu[i]);
    graph_ok = graph_ok && x_grad && bias_grad &&
        close_enough(x_grad[0], 1) && close_enough(x_grad[1], 1) &&
        close_enough(x_grad[2], 1) && close_enough(x_grad[3], 0) &&
        close_enough(bias_grad[0], 2) && close_enough(bias_grad[1], 1);
    tc_tensor_nd_free(loss0);
    tc_tensor_nd_free(relu);
    tc_tensor_nd_free(sum0);
    tc_tensor_nd_free(bias);
    tc_tensor_nd_free(x);
    if (!graph_ok) return 1;
    const size_t image_shape[] = {1, 1, 3, 3}, kernel_shape[] = {1, 1, 2, 2};
    const float image_values[] = {1,2,3,4,5,6,7,8,9};
    const float kernel_values[] = {1,0,0,1};
    const float conv_expected[] = {6,8,12,14};
    const float image_grad_expected[] = {1,1,0,1,2,1,0,1,1};
    const float kernel_grad_expected[] = {12,16,24,28};
    tc_tensor_nd *image = tc_tensor_nd_from_data(image_shape,4,image_values,1);
    tc_tensor_nd *kernel = tc_tensor_nd_from_data(kernel_shape,4,kernel_values,1);
    tc_tensor_nd *conv = tc_tensor_nd_conv2d(image,kernel);
    tc_tensor_nd *conv_loss = tc_tensor_nd_sum(conv);
    int conv_ok = image && kernel && conv && conv_loss &&
                  tc_tensor_nd_backward(conv_loss);
    const float *conv_values = tc_tensor_nd_const_data(conv);
    const float *image_grad = tc_tensor_nd_grad(image);
    const float *kernel_grad = tc_tensor_nd_grad(kernel);
    for (size_t i=0;conv_ok&&i<4;++i)
        conv_ok=close_enough(conv_values[i],conv_expected[i]);
    for (size_t i=0;conv_ok&&i<9;++i)
        conv_ok=close_enough(image_grad[i],image_grad_expected[i]);
    for (size_t i=0;conv_ok&&i<4;++i)
        conv_ok=close_enough(kernel_grad[i],kernel_grad_expected[i]);
    tc_tensor_nd_free(conv_loss);tc_tensor_nd_free(conv);
    tc_tensor_nd_free(kernel);tc_tensor_nd_free(image);
    if (!conv_ok) return 1;
    const size_t pool_shape[] = {1,1,2,2};
    const float pool_values[] = {1,4,3,2};
    tc_tensor_nd *pool_input = tc_tensor_nd_from_data(pool_shape,4,pool_values,1);
    tc_tensor_nd *pool = tc_tensor_nd_maxpool2d(pool_input,2,2);
    tc_tensor_nd *pool_loss = tc_tensor_nd_sum(pool);
    int pool_ok = pool_input && pool && pool_loss &&
                  tc_tensor_nd_backward(pool_loss) &&
                  close_enough(tc_tensor_nd_const_data(pool)[0],4.0f);
    const float *pool_grad=tc_tensor_nd_grad(pool_input);
    pool_ok=pool_ok&&pool_grad&&close_enough(pool_grad[0],0)&&
        close_enough(pool_grad[1],1)&&close_enough(pool_grad[2],0)&&
        close_enough(pool_grad[3],0);
    tc_tensor_nd_free(pool_loss);tc_tensor_nd_free(pool);
    tc_tensor_nd_free(pool_input);
    if (!pool_ok) return 1;
    const size_t class_shape[] = {2,2};
    const float logits_values[] = {1,0,0,1};
    const float target_values[] = {1,0,0,1};
    tc_tensor_nd *logits=tc_tensor_nd_from_data(class_shape,2,logits_values,1);
    tc_tensor_nd *target=tc_tensor_nd_from_data(class_shape,2,target_values,0);
    tc_tensor_nd *prob=tc_tensor_nd_softmax(logits);
    tc_tensor_nd *ce=tc_tensor_nd_cross_entropy(prob,target);
    int ce_ok=logits&&target&&prob&&ce&&tc_tensor_nd_backward(ce);
    ce_ok=ce_ok&&fabsf(tc_tensor_nd_const_data(ce)[0]-0.3132617f)<1e-4f;
    const float *logit_grad=tc_tensor_nd_grad(logits);
    ce_ok=ce_ok&&logit_grad&&close_enough(logit_grad[0],-0.1344707f)&&
        close_enough(logit_grad[1],0.1344707f)&&
        close_enough(logit_grad[2],0.1344707f)&&
        close_enough(logit_grad[3],-0.1344707f);
    tc_tensor_nd *params[]={logits};
    tc_tensor_nd_optimizer *adam=tc_tensor_nd_optimizer_adam(params,1,0.1f);
    ce_ok=ce_ok&&adam&&tc_tensor_nd_optimizer_step(adam)>0;
    const float *updated=tc_tensor_nd_const_data(logits);
    ce_ok=ce_ok&&updated&&updated[0]>logits_values[0]&&updated[1]<logits_values[1];
    tc_tensor_nd_optimizer_free(adam);
    tc_tensor_nd_free(ce);tc_tensor_nd_free(prob);
    tc_tensor_nd_free(target);tc_tensor_nd_free(logits);
    if(!ce_ok)return 1;
    const size_t a_shape[] = {2, 3}, b_shape[] = {3, 2};
    const float av[] = {1, 2, 3, 4, 5, 6};
    const float bv[] = {7, 8, 9, 10, 11, 12};
    const float expected[] = {58, 64, 139, 154};
    tc_tensor_nd *a = tc_tensor_nd_from_data(a_shape, 2, av, 1);
    tc_tensor_nd *b = tc_tensor_nd_from_data(b_shape, 2, bv, 1);
    tc_tensor_nd *out = tc_tensor_nd_matmul(a, b);
    tc_tensor_nd *loss = tc_tensor_nd_sum(out);
    int ok = a && b && out && loss && tc_tensor_nd_backward(loss);
    const float *values = tc_tensor_nd_const_data(out);
    for (size_t i = 0; ok && i < 4; ++i)
        ok = close_enough(values[i], expected[i]);
    const float *da = tc_tensor_nd_grad(a);
    ok = ok && da && close_enough(da[0], 15.0f) &&
         close_enough(da[1], 19.0f) && close_enough(da[2], 23.0f);
    tc_tensor_nd_free(loss);
    tc_tensor_nd_free(out);
    tc_tensor_nd_free(b);
    tc_tensor_nd_free(a);
    puts(ok ? "Metal matmul/autograd: ok" : "Metal matmul/autograd: failed");
    return ok ? 0 : 1;
#endif
}
