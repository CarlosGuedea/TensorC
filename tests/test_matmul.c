#include <tensorc/tensorc.h>

#include <math.h>
#include <stdio.h>

int main(void) {
    tc_tensor a = tc_tensor_create(2, 3);
    tc_tensor b = tc_tensor_create(3, 2);
    const float a_values[] = {1, 2, 3, 4, 5, 6};
    const float b_values[] = {7, 8, 9, 10, 11, 12};
    for (size_t i = 0; i < 6; ++i) a.data[i] = a_values[i];
    for (size_t i = 0; i < 6; ++i) b.data[i] = b_values[i];

    tc_tensor result = tc_tensor_matmul(&a, &b);
    const float expected[] = {58, 64, 139, 154};
    int ok = result.data && result.rows == 2 && result.cols == 2;
    for (size_t i = 0; ok && i < 4; ++i)
        ok = fabsf(result.data[i] - expected[i]) < 1e-5f;
    printf("matmul: %s\n", ok ? "ok" : "error");

    tc_tensor_free(&result);
    tc_tensor_free(&b);
    tc_tensor_free(&a);
    return ok ? 0 : 1;
}
