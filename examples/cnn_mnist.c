/*
 * Clasificador convolucional entrenable en C puro (sin dependencias).
 *
 * Uso:
 *   tensorc_cnn_mnist                                                -> sintético
 *   tensorc_cnn_mnist train-images.idx3-ubyte train-labels.idx1-ubyte \
 *                   test-images.idx3-ubyte test-labels.idx1-ubyte   -> MNIST real
 *
 * Red: conv(3x3) -> ReLU -> maxpool(2) -> [conv(3x3) -> ReLU -> maxpool(2)]
 *      -> aplanar -> matmul + sesgo -> MSE contra one-hot.
 * El entrenamiento usa el autograd de tc_tensor_nd y pasos SGD.
 */
#include <tensorc/tensorc.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* --- RNG xorshift32 determinista --- */
static unsigned int rng_state = 1u;
static void rng_seed(unsigned int seed) { rng_state = seed ? seed : 1u; }
static float rng_uniform(void) {
    unsigned int x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return (float)x / 4294967296.0f;
}

/* --- Dataset --- */
typedef struct {
    size_t count, channels, height, width, classes;
    float *images; /* [count, channels*height*width] en [0,1] */
    unsigned char *labels;
} dataset;

static void dataset_free(dataset *ds) {
    free(ds->images);
    free(ds->labels);
    memset(ds, 0, sizeof(*ds));
}

/*
 * Patrones prototipo fijos (semilla 7) + ruido con semilla configurable,
 * para poder generar conjuntos de entrenamiento y de prueba distintos
 * pero con las mismas clases. Corre sin descargar nada.
 */
static int dataset_synthetic(dataset *ds, size_t count, unsigned int noise_seed) {
    const size_t side = 8, pixels = side * side, classes = 10;
    float prototypes[10][64];
    rng_seed(7u);
    for (size_t c = 0; c < classes; ++c)
        for (size_t p = 0; p < pixels; ++p)
            prototypes[c][p] = rng_uniform();

    ds->count = count;
    ds->channels = 1;
    ds->height = side;
    ds->width = side;
    ds->classes = classes;
    ds->images = (float *)malloc(count * pixels * sizeof(float));
    ds->labels = (unsigned char *)malloc(count);
    if (!ds->images || !ds->labels) {
        dataset_free(ds);
        return 0;
    }
    rng_seed(noise_seed);
    const size_t per_class = count / classes;
    size_t index = 0;
    for (size_t c = 0; c < classes; ++c)
        for (size_t i = 0; i < per_class; ++i, ++index) {
            ds->labels[index] = (unsigned char)c;
            for (size_t p = 0; p < pixels; ++p) {
                const float noise = (rng_uniform() - 0.5f) * 0.5f;
                ds->images[index * pixels + p] =
                    fminf(fmaxf(prototypes[c][p] * 0.9f + noise, 0.0f), 1.0f);
            }
        }
    ds->count = index;
    return 1;
}

static uint32_t read_be32(FILE *file) {
    unsigned char bytes[4];
    if (fread(bytes, 1, 4, file) != 4) return 0;
    return ((uint32_t)bytes[0] << 24) | ((uint32_t)bytes[1] << 16) |
           ((uint32_t)bytes[2] << 8) | (uint32_t)bytes[3];
}

static int dataset_load_idx(const char *images_path, const char *labels_path,
                            dataset *ds, size_t max_count) {
    FILE *images = fopen(images_path, "rb");
    FILE *labels = fopen(labels_path, "rb");
    if (!images || !labels) {
        if (images) fclose(images);
        if (labels) fclose(labels);
        return 0;
    }
    if (read_be32(images) != 2051 || read_be32(labels) != 2049) {
        fclose(images);
        fclose(labels);
        return 0;
    }
    const uint32_t count = read_be32(images);
    const uint32_t rows = read_be32(images);
    const uint32_t cols = read_be32(images);
    if (read_be32(labels) != count || rows == 0 || cols == 0) {
        fclose(images);
        fclose(labels);
        return 0;
    }
    ds->count = count < max_count ? count : max_count;
    ds->channels = 1;
    ds->height = rows;
    ds->width = cols;
    ds->classes = 10;
    ds->images = (float *)malloc(ds->count * rows * cols * sizeof(float));
    ds->labels = (unsigned char *)malloc(ds->count);
    unsigned char *raw = (unsigned char *)malloc(rows * cols);
    if (!ds->images || !ds->labels || !raw) {
        free(raw);
        fclose(images);
        fclose(labels);
        dataset_free(ds);
        return 0;
    }
    for (size_t i = 0; i < ds->count; ++i) {
        const int label = fgetc(labels);
        if (label == EOF || fread(raw, 1, rows * cols, images) != rows * cols) {
            free(raw);
            fclose(images);
            fclose(labels);
            dataset_free(ds);
            return 0;
        }
        ds->labels[i] = (unsigned char)label;
        for (size_t p = 0; p < rows * cols; ++p)
            ds->images[i * rows * cols + p] = (float)raw[p] / 255.0f;
    }
    free(raw);
    fclose(images);
    fclose(labels);
    return 1;
}

/* --- Red convolucional --- */
typedef struct {
    tc_conv2d_layer *conv1;
    tc_conv2d_layer *conv2; /* NULL en la configuración pequeña */
    tc_tensor_nd *weights;  /* [features, clases] */
    tc_tensor_nd *bias;     /* [1, clases] */
    size_t classes;
} cnn;

static int cnn_create(cnn *net, size_t in_channels, size_t conv1_filters,
                      size_t conv2_filters, int two_conv_layers,
                      size_t features, size_t classes, unsigned int seed) {
    memset(net, 0, sizeof(*net));
    net->classes = classes;
    const tc_conv2d_options options = {1, 1, 1, 1};
    net->conv1 = tc_conv2d_layer_create(in_channels, conv1_filters, 3, 3,
                                        options, seed);
    net->conv2 = two_conv_layers
        ? tc_conv2d_layer_create(conv1_filters, conv2_filters, 3, 3, options,
                                 seed + 1u)
        : NULL;
    const size_t weight_shape[] = {features, classes};
    const size_t bias_shape[] = {1, classes};
    net->weights = tc_tensor_nd_create(weight_shape, 2, 1);
    net->bias = tc_tensor_nd_create(bias_shape, 2, 1);
    if (!net->conv1 || (two_conv_layers && !net->conv2) || !net->weights ||
        !net->bias)
        return 0;
    const float limit = sqrtf(6.0f / ((float)features + (float)classes));
    rng_seed(seed + 2u);
    float *w = tc_tensor_nd_data(net->weights);
    for (size_t i = 0; i < tc_tensor_nd_numel(net->weights); ++i)
        w[i] = (2.0f * rng_uniform() - 1.0f) * limit;
    return 1;
}

static void cnn_free(cnn *net) {
    tc_conv2d_layer_free(net->conv1);
    tc_conv2d_layer_free(net->conv2);
    tc_tensor_nd_free(net->weights);
    tc_tensor_nd_free(net->bias);
    memset(net, 0, sizeof(*net));
}

static tc_tensor_nd *cnn_forward(cnn *net, const tc_tensor_nd *input) {
    tc_tensor_nd *x = tc_conv2d_layer_forward(net->conv1, input);
    tc_tensor_nd *next = x ? tc_tensor_nd_relu(x) : NULL;
    tc_tensor_nd_free(x);
    x = next ? tc_tensor_nd_maxpool2d(next, 2, 2) : NULL;
    tc_tensor_nd_free(next);
    if (net->conv2) {
        next = tc_conv2d_layer_forward(net->conv2, x);
        tc_tensor_nd_free(x);
        x = next ? tc_tensor_nd_relu(next) : NULL;
        tc_tensor_nd_free(next);
        next = x ? tc_tensor_nd_maxpool2d(x, 2, 2) : NULL;
        tc_tensor_nd_free(x);
        x = next;
    }
    if (!x) return NULL;
    const size_t batch = tc_tensor_nd_shape(x)[0];
    const size_t features = tc_tensor_nd_numel(x) / batch;
    const size_t flat_shape[] = {batch, features};
    tc_tensor_nd *flat = tc_tensor_nd_reshape_to(x, flat_shape, 2);
    tc_tensor_nd_free(x);
    tc_tensor_nd *logits = flat ? tc_tensor_nd_matmul(flat, net->weights) : NULL;
    tc_tensor_nd_free(flat);
    tc_tensor_nd *out = logits ? tc_tensor_nd_add(logits, net->bias) : NULL;
    tc_tensor_nd_free(logits);
    return out;
}

static void cnn_step(cnn *net, float learning_rate) {
    tc_tensor_nd_sgd_step(tc_conv2d_layer_kernel(net->conv1), learning_rate);
    tc_tensor_nd_sgd_step(tc_conv2d_layer_bias(net->conv1), learning_rate);
    if (net->conv2) {
        tc_tensor_nd_sgd_step(tc_conv2d_layer_kernel(net->conv2), learning_rate);
        tc_tensor_nd_sgd_step(tc_conv2d_layer_bias(net->conv2), learning_rate);
    }
    tc_tensor_nd_sgd_step(net->weights, learning_rate);
    tc_tensor_nd_sgd_step(net->bias, learning_rate);
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static float *build_targets(const dataset *ds, size_t begin, size_t batch) {
    float *targets = (float *)calloc(batch * ds->classes, sizeof(float));
    if (!targets) return NULL;
    for (size_t i = 0; i < batch; ++i)
        targets[i * ds->classes + ds->labels[begin + i]] = 1.0f;
    return targets;
}

static int run_batch(cnn *net, const dataset *ds, size_t begin, size_t batch,
                     const float *targets, int train, float learning_rate,
                     float *loss_out) {
    const size_t sample_size = ds->channels * ds->height * ds->width;
    const size_t shape[] = {batch, ds->channels, ds->height, ds->width};
    const size_t target_shape[] = {batch, ds->classes};
    tc_tensor_nd *input = tc_tensor_nd_from_data(shape, 4,
                                                 ds->images + begin * sample_size,
                                                 train);
    tc_tensor_nd *target = tc_tensor_nd_from_data(target_shape, 2, targets, 0);
    tc_tensor_nd *logits = cnn_forward(net, input);
    tc_tensor_nd *loss = logits ? tc_tensor_nd_mse_loss(logits, target) : NULL;
    const int ok = input && target && logits && loss &&
                   (!train || tc_tensor_nd_backward(loss));
    if (ok && loss_out) *loss_out = tc_tensor_nd_const_data(loss)[0];
    if (ok && train) cnn_step(net, learning_rate);
    tc_tensor_nd_free(loss);
    tc_tensor_nd_free(logits);
    tc_tensor_nd_free(target);
    tc_tensor_nd_free(input);
    return ok;
}

static int train_network(cnn *net, const dataset *ds, size_t batch_size,
                         size_t epochs, float learning_rate) {
    const size_t batches = ds->count / batch_size;
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        const double start = now_ms();
        float total = 0.0f;
        for (size_t b = 0; b < batches; ++b) {
            const size_t begin = b * batch_size;
            float *targets = build_targets(ds, begin, batch_size);
            float loss = 0.0f;
            const int ok = targets &&
                           run_batch(net, ds, begin, batch_size, targets, 1,
                                     learning_rate, &loss);
            free(targets);
            if (!ok) return 0;
            total += loss;
        }
        printf("  época %zu: pérdida %.5f (%.1f ms)\n", epoch + 1,
               total / (float)batches, now_ms() - start);
    }
    return 1;
}

static int accuracy(cnn *net, const dataset *ds, size_t batch_size) {
    const size_t batches = ds->count / batch_size;
    size_t correct = 0, total = 0;
    for (size_t b = 0; b < batches; ++b) {
        const size_t begin = b * batch_size;
        float *targets = build_targets(ds, begin, batch_size);
        const size_t shape[] = {batch_size, ds->channels, ds->height, ds->width};
        tc_tensor_nd *input = targets ? tc_tensor_nd_from_data(
            shape, 4, ds->images + begin * ds->channels * ds->height * ds->width,
            0) : NULL;
        tc_tensor_nd *logits = input ? cnn_forward(net, input) : NULL;
        const float *scores = tc_tensor_nd_const_data(logits);
        if (scores) {
            for (size_t i = 0; i < batch_size; ++i) {
                size_t best = 0;
                for (size_t c = 1; c < ds->classes; ++c)
                    if (scores[i * ds->classes + c] >
                        scores[i * ds->classes + best])
                        best = c;
                if (best == ds->labels[begin + i]) ++correct;
                ++total;
            }
        }
        tc_tensor_nd_free(logits);
        tc_tensor_nd_free(input);
        free(targets);
    }
    return total ? (int)(100.0 * (double)correct / (double)total + 0.5) : -1;
}

int main(int argc, char **argv) {
    dataset train_set = {0}, test_set = {0};
    size_t features, conv1_filters, conv2_filters, batch_size, epochs;
    float learning_rate;
    int real_mnist = 0;

    if (argc == 5) {
        real_mnist = dataset_load_idx(argv[1], argv[2], &train_set, 8192) &&
                     dataset_load_idx(argv[3], argv[4], &test_set, 2048);
        if (!real_mnist)
            printf("No se pudieron leer los archivos IDX; uso sintético.\n");
    }
    if (real_mnist) {
        conv1_filters = 8;
        conv2_filters = 16;
        features = conv2_filters * 7 * 7;
        batch_size = 64;
        epochs = 4;
        learning_rate = 0.03f;
        printf("MNIST: %zu entrenamiento, %zu prueba\n", train_set.count,
               test_set.count);
    } else {
        dataset_free(&train_set);
        dataset_free(&test_set);
        if (!dataset_synthetic(&train_set, 800, 11u) ||
            !dataset_synthetic(&test_set, 200, 99u))
            return 2;
        conv1_filters = 4;
        conv2_filters = 0;
        features = conv1_filters * 4 * 4;
        batch_size = 40;
        epochs = 30;
        learning_rate = 0.1f;
        printf("Dataset sintético: %zu entrenamiento, %zu prueba\n",
               train_set.count, test_set.count);
    }

    cnn net;
    if (!cnn_create(&net, 1, conv1_filters, conv2_filters, real_mnist,
                    features, train_set.classes, 123u)) {
        printf("No se pudo crear la red\n");
        dataset_free(&train_set);
        dataset_free(&test_set);
        return 2;
    }
    const double start = now_ms();
    const int trained = train_network(&net, &train_set, batch_size, epochs,
                                      learning_rate);
    const double train_ms = now_ms() - start;
    const int acc = trained ? accuracy(&net, &test_set, batch_size) : -1;
    printf("Entrenamiento: %.1f ms total | Precisión en prueba: %d%%\n",
           train_ms, acc);

    cnn_free(&net);
    dataset_free(&train_set);
    dataset_free(&test_set);
    return trained && acc >= 0 ? 0 : 1;
}
