/*
 * Clasificador de lugares para estudiar usando únicamente TensorC/C11.
 *
 * El programa recibe archivos binarios preparados desde las fotografías:
 *   study_classifier train.bin val.bin test.bin modelo.tck test.csv
 *   study_classifier train.bin val.bin test.bin modelo.tck test.csv
 *                    uncertain.bin uncertain.csv
 *
 * Red: Conv2D(3->8, 3x3, padding 1) -> ReLU -> MaxPool(2)
 *      -> Conv2D(8->16, 3x3, padding 1) -> ReLU -> MaxPool(2)
 *      -> Dense(1024->2) -> Softmax.
 *
 * El binario de entrada es producido por el preprocesador del experimento:
 * magic TCSD, count, channels, height, width, metadata_dim, labels uint8,
 * imagenes float32 y metadatos float32.
 */
#include <tensorc/tensorc.h>
#include "../tensorc_time.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { CLASS_NO_APTO = 0, CLASS_APTO = 1, UNKNOWN_LABEL = 255 };
static const uint32_t DATA_MAGIC = 0x54435344u; /* TCSD */

typedef struct {
    uint32_t count, channels, height, width, metadata_dim;
    unsigned char *labels;
    float *images;
    float *metadata;
} dataset;

typedef struct {
    tc_conv2d_layer *conv1;
    tc_conv2d_layer *conv2;
    tc_dense_layer *image_projection;
    tc_dense_layer *metadata_projection;
    tc_dense_layer *classifier;
    tc_tensor_nd_optimizer *optimizer;
} network;

static unsigned int rng_state = 123u;

static void rng_seed(unsigned int seed) { rng_state = seed ? seed : 1u; }

static unsigned int rng_next(void) {
    unsigned int x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return x;
}

static double now_ms(void) {
    return tc_monotonic_ms();
}

static void dataset_free(dataset *ds) {
    free(ds->labels);
    free(ds->images);
    free(ds->metadata);
    memset(ds, 0, sizeof(*ds));
}

static int read_u32(FILE *f, uint32_t *out) {
    return fread(out, sizeof(*out), 1, f) == 1;
}

static int dataset_load(const char *path, dataset *ds) {
    memset(ds, 0, sizeof(*ds));
    FILE *f = fopen(path, "rb");
    uint32_t magic = 0;
    if (!f || !read_u32(f, &magic) || magic != DATA_MAGIC ||
        !read_u32(f, &ds->count) || !read_u32(f, &ds->channels) ||
        !read_u32(f, &ds->height) || !read_u32(f, &ds->width) ||
        !read_u32(f, &ds->metadata_dim) ||
        ds->count == 0 || ds->channels == 0 || ds->height == 0 ||
        ds->width == 0 || ds->channels > 4 || ds->metadata_dim == 0 ||
        ds->metadata_dim > 64) {
        if (f) fclose(f);
        return 0;
    }
    const size_t pixels = (size_t)ds->channels * ds->height * ds->width;
    ds->labels = (unsigned char *)malloc(ds->count);
    ds->images = (float *)malloc((size_t)ds->count * pixels * sizeof(float));
    ds->metadata = (float *)malloc((size_t)ds->count * ds->metadata_dim *
                                   sizeof(float));
    if (!ds->labels || !ds->images || !ds->metadata ||
        fread(ds->labels, 1, ds->count, f) != ds->count ||
        fread(ds->images, sizeof(float), (size_t)ds->count * pixels, f) !=
            (size_t)ds->count * pixels ||
        fread(ds->metadata, sizeof(float),
              (size_t)ds->count * ds->metadata_dim, f) !=
            (size_t)ds->count * ds->metadata_dim) {
        fclose(f);
        dataset_free(ds);
        return 0;
    }
    fclose(f);
    return 1;
}

static int network_create(network *net, size_t channels, size_t height,
                           size_t width, size_t metadata_dim) {
    memset(net, 0, sizeof(*net));
    const tc_conv2d_options options = {1, 1, 1, 1};
    net->conv1 = tc_conv2d_layer_create(channels, 8, 3, 3, options, 101u);
    net->conv2 = tc_conv2d_layer_create(8, 16, 3, 3, options, 202u);
    net->image_projection = tc_dense_layer_create(
        16 * (height / 4) * (width / 4), 16, 303u);
    net->metadata_projection = tc_dense_layer_create(metadata_dim, 8, 404u);
    net->classifier = tc_dense_layer_create(24, 2, 505u);
    if (!net->conv1 || !net->conv2 || !net->image_projection ||
        !net->metadata_projection || !net->classifier)
        return 0;

    tc_tensor_nd *params[] = {
        tc_conv2d_layer_kernel(net->conv1), tc_conv2d_layer_bias(net->conv1),
        tc_conv2d_layer_kernel(net->conv2), tc_conv2d_layer_bias(net->conv2),
        tc_dense_layer_weight(net->image_projection),
        tc_dense_layer_bias(net->image_projection),
        tc_dense_layer_weight(net->metadata_projection),
        tc_dense_layer_bias(net->metadata_projection),
        tc_dense_layer_weight(net->classifier),
        tc_dense_layer_bias(net->classifier),
    };
    net->optimizer = tc_tensor_nd_optimizer_adam(params, 10, 0.001f);
    return net->optimizer != NULL;
}

static void network_free(network *net) {
    tc_tensor_nd_optimizer_free(net->optimizer);
    tc_conv2d_layer_free(net->conv1);
    tc_conv2d_layer_free(net->conv2);
    tc_dense_layer_free(net->image_projection);
    tc_dense_layer_free(net->metadata_projection);
    tc_dense_layer_free(net->classifier);
    memset(net, 0, sizeof(*net));
}

static tc_tensor_nd *network_forward(network *net, const tc_tensor_nd *input,
                                     const tc_tensor_nd *metadata) {
    tc_tensor_nd *x = tc_conv2d_layer_forward(net->conv1, input);
    tc_tensor_nd *next = x ? tc_tensor_nd_relu(x) : NULL;
    tc_tensor_nd_free(x);
    x = next ? tc_tensor_nd_maxpool2d(next, 2, 2) : NULL;
    tc_tensor_nd_free(next);

    next = x ? tc_conv2d_layer_forward(net->conv2, x) : NULL;
    tc_tensor_nd_free(x);
    x = next ? tc_tensor_nd_relu(next) : NULL;
    tc_tensor_nd_free(next);
    next = x ? tc_tensor_nd_maxpool2d(x, 2, 2) : NULL;
    tc_tensor_nd_free(x);
    if (!next) return NULL;

    const size_t batch = tc_tensor_nd_shape(next)[0];
    const size_t features = tc_tensor_nd_numel(next) / batch;
    const size_t flat_shape[] = {batch, features};
    x = tc_tensor_nd_reshape_to(next, flat_shape, 2);
    tc_tensor_nd_free(next);
    tc_tensor_nd *image = x ? tc_dense_layer_forward(net->image_projection, x)
                            : NULL;
    tc_tensor_nd_free(x);
    x = image ? tc_tensor_nd_relu(image) : NULL;
    tc_tensor_nd_free(image);
    tc_tensor_nd *meta = metadata
                             ? tc_dense_layer_forward(net->metadata_projection,
                                                      metadata)
                             : NULL;
    next = meta ? tc_tensor_nd_relu(meta) : NULL;
    tc_tensor_nd_free(meta);
    if (!x || !next) {
        tc_tensor_nd_free(x);
        tc_tensor_nd_free(next);
        return NULL;
    }
    const tc_tensor_nd *branches[] = {x, next};
    tc_tensor_nd *joined = tc_tensor_nd_concat(branches, 2, 1);
    tc_tensor_nd_free(x);
    tc_tensor_nd_free(next);
    tc_tensor_nd *out = joined ? tc_dense_layer_forward(net->classifier, joined)
                               : NULL;
    tc_tensor_nd_free(joined);
    return out;
}

static int make_batch(const dataset *ds, const size_t *indices, size_t begin,
                      size_t batch, float **images_out,
                      unsigned char **labels_out, float **metadata_out) {
    const size_t sample = (size_t)ds->channels * ds->height * ds->width;
    float *images = (float *)malloc(batch * sample * sizeof(float));
    unsigned char *labels = (unsigned char *)malloc(batch);
    float *metadata = (float *)malloc(batch * ds->metadata_dim * sizeof(float));
    if (!images || !labels || !metadata) {
        free(images);
        free(labels);
        free(metadata);
        return 0;
    }
    for (size_t i = 0; i < batch; ++i) {
        const size_t source = indices ? indices[begin + i] : begin + i;
        memcpy(images + i * sample, ds->images + source * sample,
               sample * sizeof(float));
        memcpy(metadata + i * ds->metadata_dim,
               ds->metadata + source * ds->metadata_dim,
               ds->metadata_dim * sizeof(float));
        labels[i] = ds->labels[source];
    }
    *images_out = images;
    *labels_out = labels;
    *metadata_out = metadata;
    return 1;
}

static float *one_hot(const unsigned char *labels, size_t batch) {
    float *targets = (float *)calloc(batch * 2, sizeof(float));
    if (!targets) return NULL;
    for (size_t i = 0; i < batch; ++i)
        targets[i * 2 + (labels[i] == CLASS_APTO ? CLASS_APTO : CLASS_NO_APTO)] =
            1.0f;
    return targets;
}

static int train_batch(network *net, const dataset *ds, const size_t *indices,
                       size_t begin, size_t batch, float *loss_out) {
    float *images = NULL;
    unsigned char *labels = NULL;
    float *metadata = NULL;
    float *targets = NULL;
    const size_t input_shape[] = {batch, ds->channels, ds->height, ds->width};
    const size_t metadata_shape[] = {batch, ds->metadata_dim};
    const size_t target_shape[] = {batch, 2};
    if (!make_batch(ds, indices, begin, batch, &images, &labels, &metadata))
        return 0;
    targets = one_hot(labels, batch);
    tc_tensor_nd *input = targets ? tc_tensor_nd_from_data(
        input_shape, 4, images, 1) : NULL;
    tc_tensor_nd *metadata_tensor = targets ? tc_tensor_nd_from_data(
        metadata_shape, 2, metadata, 1) : NULL;
    tc_tensor_nd *target = targets ? tc_tensor_nd_from_data(
        target_shape, 2, targets, 0) : NULL;
    tc_tensor_nd *logits = input && metadata_tensor
                               ? network_forward(net, input, metadata_tensor)
                               : NULL;
    tc_tensor_nd *probs = logits ? tc_tensor_nd_softmax(logits) : NULL;
    tc_tensor_nd *loss = probs ? tc_tensor_nd_cross_entropy(probs, target) : NULL;
    int ok = input && target && logits && probs && loss &&
             tc_tensor_nd_backward(loss);
    if (ok && loss_out) *loss_out = tc_tensor_nd_const_data(loss)[0];
    if (ok) ok = tc_tensor_nd_optimizer_step(net->optimizer) > 0;
    tc_tensor_nd_free(loss);
    tc_tensor_nd_free(probs);
    tc_tensor_nd_free(logits);
    tc_tensor_nd_free(target);
    tc_tensor_nd_free(metadata_tensor);
    tc_tensor_nd_free(input);
    free(targets);
    free(labels);
    free(metadata);
    free(images);
    return ok;
}

typedef struct {
    float loss;
    size_t total, correct;
    size_t confusion[2][2];
} metrics;

static int evaluate(network *net, const dataset *ds, size_t batch_size,
                    metrics *out, const char *csv_path) {
    memset(out, 0, sizeof(*out));
    FILE *csv = csv_path ? fopen(csv_path, "w") : NULL;
    if (csv_path && !csv) return 0;
    if (csv) fprintf(csv, "indice,real,prediccion,prob_apto\n");
    for (size_t begin = 0; begin < ds->count; begin += batch_size) {
        const size_t batch = (ds->count - begin < batch_size)
                                 ? ds->count - begin
                                 : batch_size;
        float *images = NULL;
        unsigned char *labels = NULL;
        float *metadata = NULL;
        if (!make_batch(ds, NULL, begin, batch, &images, &labels, &metadata)) {
            if (csv) fclose(csv);
            return 0;
        }
        const size_t input_shape[] = {batch, ds->channels, ds->height, ds->width};
        const size_t metadata_shape[] = {batch, ds->metadata_dim};
        tc_tensor_nd *input = tc_tensor_nd_from_data(input_shape, 4, images, 0);
        tc_tensor_nd *metadata_tensor = tc_tensor_nd_from_data(
            metadata_shape, 2, metadata, 0);
        tc_tensor_nd *logits = input && metadata_tensor
                                   ? network_forward(net, input, metadata_tensor)
                                   : NULL;
        tc_tensor_nd *probs = logits ? tc_tensor_nd_softmax(logits) : NULL;
        if (!input || !logits || !probs) {
            tc_tensor_nd_free(probs);
            tc_tensor_nd_free(logits);
            tc_tensor_nd_free(input);
            free(labels);
            tc_tensor_nd_free(metadata_tensor);
            free(metadata);
            free(images);
            if (csv) fclose(csv);
            return 0;
        }
        const float *scores = tc_tensor_nd_const_data(probs);
        for (size_t i = 0; i < batch; ++i) {
            const int pred = scores[i * 2 + 1] >= scores[i * 2] ? 1 : 0;
            const unsigned char real = labels[i];
            if (real == CLASS_APTO || real == CLASS_NO_APTO) {
                const int actual = real == CLASS_APTO ? 1 : 0;
                out->loss += -logf(fmaxf(scores[i * 2 + actual], 1e-7f));
                out->confusion[actual][pred]++;
                out->total++;
                out->correct += actual == pred;
            }
            if (csv)
                fprintf(csv, "%zu,%s,%s,%.7f\n", begin + i,
                        real == CLASS_APTO ? "apto"
                                           : real == CLASS_NO_APTO ? "no_apto"
                                                                   : "incierto",
                        pred ? "apto" : "no_apto", scores[i * 2 + 1]);
        }
        tc_tensor_nd_free(probs);
        tc_tensor_nd_free(logits);
        tc_tensor_nd_free(metadata_tensor);
        tc_tensor_nd_free(input);
        free(labels);
        free(metadata);
        free(images);
    }
    if (csv) fclose(csv);
    if (out->total) out->loss /= (float)out->total;
    return 1;
}

static void print_metrics(const char *name, const metrics *m) {
    const double tp = (double)m->confusion[1][1];
    const double fp = (double)m->confusion[0][1];
    const double fn = (double)m->confusion[1][0];
    const double precision = tp + fp ? tp / (tp + fp) : 0.0;
    const double recall = tp + fn ? tp / (tp + fn) : 0.0;
    const double f1 = precision + recall
                          ? 2.0 * precision * recall / (precision + recall)
                          : 0.0;
    printf("%s | pérdida %.5f | exactitud %.2f%% | precision %.2f%% | "
           "recall %.2f%% | F1 %.2f%% | matriz [[%zu,%zu],[%zu,%zu]]\n",
           name, m->loss, m->total ? 100.0 * (double)m->correct / m->total : 0.0,
           100.0 * precision, 100.0 * recall, 100.0 * f1,
           m->confusion[0][0], m->confusion[0][1], m->confusion[1][0],
           m->confusion[1][1]);
}

static int save_model(network *net, const char *path) {
    const tc_tensor_nd *params[] = {
        tc_conv2d_layer_kernel(net->conv1), tc_conv2d_layer_bias(net->conv1),
        tc_conv2d_layer_kernel(net->conv2), tc_conv2d_layer_bias(net->conv2),
        tc_dense_layer_weight(net->image_projection),
        tc_dense_layer_bias(net->image_projection),
        tc_dense_layer_weight(net->metadata_projection),
        tc_dense_layer_bias(net->metadata_projection),
        tc_dense_layer_weight(net->classifier),
        tc_dense_layer_bias(net->classifier),
    };
    return tc_save_checkpoint(params, 10, path);
}

static int load_model(network *net, const char *path) {
    size_t count = 0;
    tc_tensor_nd **saved = tc_load_checkpoint(path, &count, 0);
    if (!saved || count != 10) {
        tc_free_checkpoint(saved, count);
        return 0;
    }
    tc_tensor_nd *params[] = {
        tc_conv2d_layer_kernel(net->conv1), tc_conv2d_layer_bias(net->conv1),
        tc_conv2d_layer_kernel(net->conv2), tc_conv2d_layer_bias(net->conv2),
        tc_dense_layer_weight(net->image_projection),
        tc_dense_layer_bias(net->image_projection),
        tc_dense_layer_weight(net->metadata_projection),
        tc_dense_layer_bias(net->metadata_projection),
        tc_dense_layer_weight(net->classifier),
        tc_dense_layer_bias(net->classifier),
    };
    int ok = 1;
    for (size_t i = 0; i < count; ++i)
        ok = tc_tensor_nd_copy_(params[i], saved[i]) && ok;
    tc_free_checkpoint(saved, count);
    return ok;
}

int main(int argc, char **argv) {
    if (argc != 6 && argc != 8) {
        fprintf(stderr, "Uso: %s train.bin val.bin test.bin modelo.tck test.csv "
                        "[uncertain.bin uncertain.csv]\n",
                argv[0]);
        return 2;
    }
    dataset train = {0}, val = {0}, test = {0}, uncertain = {0};
    if (!dataset_load(argv[1], &train) || !dataset_load(argv[2], &val) ||
        !dataset_load(argv[3], &test) ||
        (argc == 8 && !dataset_load(argv[6], &uncertain))) {
        fprintf(stderr, "No se pudo leer uno de los datasets binarios.\n");
        dataset_free(&train); dataset_free(&val); dataset_free(&test);
        dataset_free(&uncertain);
        return 2;
    }
    if (train.channels != 3 || train.height != 32 || train.width != 32 ||
        train.metadata_dim != 11 || val.channels != train.channels ||
        test.channels != train.channels || val.metadata_dim != train.metadata_dim ||
        test.metadata_dim != train.metadata_dim) {
        fprintf(stderr, "Dimensiones inesperadas; se requieren imagenes RGB 32x32.\n");
        dataset_free(&train); dataset_free(&val); dataset_free(&test);
        dataset_free(&uncertain);
        return 2;
    }
    network net;
    if (!network_create(&net, train.channels, train.height, train.width,
                        train.metadata_dim)) {
        fprintf(stderr, "No se pudo crear la CNN TensorC.\n");
        dataset_free(&train); dataset_free(&val); dataset_free(&test);
        dataset_free(&uncertain);
        return 2;
    }
    size_t class_count[2] = {0, 0};
    for (size_t i = 0; i < train.count; ++i)
        if (train.labels[i] == CLASS_APTO)
            class_count[1]++;
        else
            class_count[0]++;
    /* Oversampling determinista: cada época ve el mismo número de ejemplos
     * de ambas clases, sin alterar validación ni prueba. */
    const size_t balanced_per_class = class_count[0] > class_count[1]
                                          ? class_count[0]
                                          : class_count[1];
    const size_t order_count = balanced_per_class * 2;
    size_t *order = (size_t *)malloc(order_count * sizeof(size_t));
    if (!order) {
        network_free(&net); dataset_free(&train); dataset_free(&val);
        dataset_free(&test); dataset_free(&uncertain);
        return 2;
    }
    size_t class_pos[2] = {0, 0};
    size_t class_indices[2][256];
    for (size_t i = 0; i < train.count; ++i) {
        const int cls = train.labels[i] == CLASS_APTO ? 1 : 0;
        if (class_pos[cls] < 256) class_indices[cls][class_pos[cls]++] = i;
    }
    for (size_t i = 0; i < balanced_per_class; ++i) {
        order[i] = class_indices[0][i % class_pos[0]];
        order[balanced_per_class + i] = class_indices[1][i % class_pos[1]];
    }
    const size_t batch_size = 16;
    const size_t epochs = 35;
    float best_val = INFINITY;
    size_t best_epoch = 0;
    rng_seed(77u);
    const double start = now_ms();
    for (size_t epoch = 1; epoch <= epochs; ++epoch) {
        for (size_t i = train.count; i > 1; --i) {
            const size_t j = rng_next() % i;
            const size_t tmp = order[i - 1]; order[i - 1] = order[j]; order[j] = tmp;
        }
        double train_loss = 0.0;
        size_t batches = 0;
        for (size_t begin = 0; begin < order_count; begin += batch_size) {
            const size_t batch = order_count - begin < batch_size
                                     ? order_count - begin
                                     : batch_size;
            float loss = 0.0f;
            if (!train_batch(&net, &train, order, begin, batch, &loss)) {
                fprintf(stderr, "Fallo en el entrenamiento, epoca %zu.\n", epoch);
                free(order); network_free(&net); dataset_free(&train);
                dataset_free(&val); dataset_free(&test); dataset_free(&uncertain);
                return 2;
            }
            train_loss += loss;
            batches++;
        }
        metrics vm;
        if (!evaluate(&net, &val, batch_size, &vm, NULL)) {
            fprintf(stderr, "Fallo al evaluar validacion.\n");
            free(order); network_free(&net); dataset_free(&train);
            dataset_free(&val); dataset_free(&test); dataset_free(&uncertain);
            return 2;
        }
        printf("epoca %02zu/%zu | train_loss %.5f | val_loss %.5f | "
               "val_acc %.2f%%\n", epoch, epochs, train_loss / (double)batches,
               vm.loss, vm.total ? 100.0 * (double)vm.correct / vm.total : 0.0);
        if (vm.loss < best_val) {
            best_val = vm.loss;
            best_epoch = epoch;
            if (!save_model(&net, argv[4]))
                fprintf(stderr, "Aviso: no se pudo guardar el checkpoint.\n");
        }
    }
    printf("Entrenamiento TensorC: %.1f ms | mejor epoca: %zu\n",
           now_ms() - start, best_epoch);
    if (!load_model(&net, argv[4]))
        fprintf(stderr, "Aviso: no se pudo recargar el mejor checkpoint; "
                        "se evaluara el ultimo estado.\n");
    metrics tm;
    if (!evaluate(&net, &test, batch_size, &tm, argv[5])) {
        fprintf(stderr, "Fallo al evaluar prueba.\n");
        free(order); network_free(&net); dataset_free(&train);
        dataset_free(&val); dataset_free(&test); dataset_free(&uncertain);
        return 2;
    }
    print_metrics("PRUEBA", &tm);
    if (argc == 8) {
        metrics um;
        if (evaluate(&net, &uncertain, batch_size, &um, argv[7]))
            printf("INCIERTOS: %zu fotografias clasificadas; resultados en %s\n",
                   (size_t)uncertain.count, argv[7]);
    }
    free(order);
    network_free(&net);
    dataset_free(&train); dataset_free(&val); dataset_free(&test);
    dataset_free(&uncertain);
    return 0;
}
