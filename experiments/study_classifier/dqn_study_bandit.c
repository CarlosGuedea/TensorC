/* Contextual-bandit demo: frozen study-place CNN features + TensorC DQN. */
#include <tensorc/tensorc.h>

#include <math.h>
#include <setjmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <jpeglib.h>

enum {
    IMAGE_SIDE = 32,
    IMAGE_CHANNELS = 3,
    VISUAL_FEATURES = 16,
    METADATA_FEATURES = 11,
    OBSERVATION_SIZE = VISUAL_FEATURES + METADATA_FEATURES,
    ACTIONS = 2,
    MAX_FIELDS = 32,
    MAX_LINE = 8192,
    MAX_PATH = 1024,
    MAX_ID = 80,
    EPISODES = 8000,
};

typedef enum { SPLIT_NONE, SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST } data_split;

typedef struct {
    char id[MAX_ID];
    char label[16];
    char photo[MAX_PATH];
    float metadata[METADATA_FEATURES];
    float observation[OBSERVATION_SIZE];
    int class_id;
    data_split split;
    int in_manifest;
} study_row;

typedef struct {
    study_row *rows;
    size_t count;
    size_t capacity;
    size_t uncertain_count;
} study_data;

typedef struct {
    tc_conv2d_layer *conv1;
    tc_conv2d_layer *conv2;
    tc_dense_layer *projection;
} image_encoder;

typedef struct {
    study_row **train_rows;
    size_t train_count;
    study_row *current;
    uint32_t rng;
} bandit_state;

typedef struct {
    struct jpeg_error_mgr base;
    jmp_buf jump;
} jpeg_error_state;

typedef struct {
    struct jpeg_decompress_struct decoder;
    jpeg_error_state error;
    unsigned char *pixels;
    int created;
} jpeg_state;

static double seconds_now(void) {
    struct timespec value;
    timespec_get(&value, TIME_UTC);
    return (double)value.tv_sec + (double)value.tv_nsec / 1e9;
}

static uint32_t random_u32(uint32_t *state) {
    uint32_t value = *state ? *state : 1u;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    *state = value;
    return value;
}

/* CSV fields in this export are single-line; quoted commas and escaped quotes are supported. */
static int csv_split(char *line, char **fields, size_t capacity) {
    char *src = line, *dst = line;
    size_t count = 0;
    while (1) {
        if (count == capacity) return -1;
        fields[count++] = dst;
        int quoted = *src == '"';
        if (quoted) ++src;
        while (*src) {
            if (quoted && *src == '"') {
                if (src[1] == '"') {
                    *dst++ = '"';
                    src += 2;
                    continue;
                }
                ++src;
                quoted = 0;
                if (*src && *src != ',' && *src != '\r' && *src != '\n')
                    return -1;
                break;
            }
            if (!quoted && (*src == ',' || *src == '\r' || *src == '\n'))
                break;
            *dst++ = *src++;
        }
        if (quoted) return -1;
        const char delimiter = *src;
        *dst++ = '\0';
        if (delimiter != ',') break;
        ++src;
    }
    return (int)count;
}

static void remove_utf8_bom(char *value) {
    const unsigned char *bytes = (const unsigned char *)value;
    if (strlen(value) >= 3 && bytes[0] == 0xef && bytes[1] == 0xbb &&
        bytes[2] == 0xbf)
        memmove(value, value + 3, strlen(value + 3) + 1);
}

static int read_csv_header(FILE *file, char *line, size_t line_size,
                           char **fields, size_t field_capacity,
                           int *field_count) {
    if (!fgets(line, (int)line_size, file)) return 0;
    if (!strchr(line, '\n') && !feof(file)) return 0;
    *field_count = csv_split(line, fields, field_capacity);
    if (*field_count < 0) return 0;
    remove_utf8_bom(fields[0]);
    return 1;
}

static int find_column(char **header, int count, const char *name) {
    for (int i = 0; i < count; ++i)
        if (strcmp(header[i], name) == 0) return i;
    return -1;
}

static int field_at(char **fields, int count, int index, const char **out) {
    if (index < 0 || index >= count) return 0;
    *out = fields[index];
    return 1;
}

static int parse_float_field(const char *text, float fallback, float *out) {
    if (!text || !*text) {
        *out = fallback;
        return 1;
    }
    char *end = NULL;
    const float value = strtof(text, &end);
    if (end == text || *end || !isfinite(value)) return 0;
    *out = value;
    return 1;
}

static int encode_metadata(char **fields, int count, const int columns[7],
                           float output[METADATA_FEATURES]) {
    const char *people, *table, *occupancy, *space;
    const char *temperature, *humidity, *sound;
    if (!field_at(fields, count, columns[0], &people) ||
        !field_at(fields, count, columns[1], &table) ||
        !field_at(fields, count, columns[2], &occupancy) ||
        !field_at(fields, count, columns[3], &space) ||
        !field_at(fields, count, columns[4], &temperature) ||
        !field_at(fields, count, columns[5], &humidity) ||
        !field_at(fields, count, columns[6], &sound))
        return 0;

    float people_value, temperature_value, humidity_value, sound_value;
    if (!parse_float_field(people, 0.0f, &people_value) || people_value < 0.0f ||
        !parse_float_field(temperature, 30.0f, &temperature_value) ||
        !parse_float_field(humidity, 0.0f, &humidity_value) ||
        !parse_float_field(sound, -40.0f, &sound_value))
        return 0;
    if (*table && strcmp(table, "True") != 0 && strcmp(table, "False") != 0)
        return 0;

    memset(output, 0, METADATA_FEATURES * sizeof(float));
    output[0] = fminf(people_value, 30.0f) / 30.0f;
    output[1] = strcmp(table, "True") == 0 ? 1.0f : 0.0f;
    if (strcmp(occupancy, "despejado") == 0) output[2] = 1.0f;
    else if (strcmp(occupancy, "medio") == 0) output[3] = 1.0f;
    else if (strcmp(occupancy, "lleno") == 0) output[4] = 1.0f;
    if (strcmp(space, "exterior") == 0) output[5] = 1.0f;
    else if (strcmp(space, "interior") == 0) output[6] = 1.0f;
    else if (strcmp(space, "mixto") == 0) output[7] = 1.0f;
    output[8] = (temperature_value - 20.0f) / 20.0f;
    output[9] = humidity_value / 100.0f;
    output[10] = (sound_value + 40.0f) / 40.0f;
    return 1;
}

static int append_row(study_data *data, const study_row *row) {
    if (data->count == data->capacity) {
        const size_t capacity = data->capacity ? data->capacity * 2 : 64;
        if (capacity < data->capacity || capacity > SIZE_MAX / sizeof(*data->rows))
            return 0;
        study_row *rows = (study_row *)realloc(data->rows,
                                               capacity * sizeof(*rows));
        if (!rows) return 0;
        data->rows = rows;
        data->capacity = capacity;
    }
    data->rows[data->count++] = *row;
    return 1;
}

static int load_data_csv(const char *path, study_data *data) {
    FILE *file = fopen(path, "r");
    char line[MAX_LINE], *fields[MAX_FIELDS];
    int columns = 0;
    if (!file) return 0;
    if (!read_csv_header(file, line, sizeof(line), fields, MAX_FIELDS, &columns)) {
        fclose(file);
        return 0;
    }
    const int id_col = find_column(fields, columns, "id");
    const int label_col = find_column(fields, columns, "study_label");
    const int photo_col = find_column(fields, columns, "foto_local");
    const int metadata_columns[7] = {
        find_column(fields, columns, "people_count"),
        find_column(fields, columns, "has_table"),
        find_column(fields, columns, "visual_occupancy"),
        find_column(fields, columns, "space_type"),
        find_column(fields, columns, "temperature_c"),
        find_column(fields, columns, "humidity_pct"),
        find_column(fields, columns, "sound_dbfs"),
    };
    if (id_col < 0 || label_col < 0 || photo_col < 0) {
        fclose(file);
        return 0;
    }
    for (size_t i = 0; i < sizeof(metadata_columns) / sizeof(metadata_columns[0]);
         ++i) {
        if (metadata_columns[i] < 0) {
            fclose(file);
            return 0;
        }
    }

    int ok = 1;
    while (fgets(line, sizeof(line), file)) {
        if (!strchr(line, '\n') && !feof(file)) { ok = 0; break; }
        const int field_count = csv_split(line, fields, MAX_FIELDS);
        const char *id, *label, *photo;
        if (field_count < 0 || !field_at(fields, field_count, id_col, &id) ||
            !field_at(fields, field_count, label_col, &label) ||
            !field_at(fields, field_count, photo_col, &photo)) {
            ok = 0;
            break;
        }
        if (strcmp(label, "incierto") == 0) {
            ++data->uncertain_count;
            continue;
        }
        if (strcmp(label, "apto") != 0 && strcmp(label, "no_apto") != 0) {
            ok = 0;
            break;
        }
        study_row row = {0};
        if (strlen(id) >= sizeof(row.id) || strlen(label) >= sizeof(row.label) ||
            strlen(photo) >= sizeof(row.photo) || !*id || !*photo ||
            strncmp(photo, "fotos/", 6) != 0 || strstr(photo, "..") ||
            !encode_metadata(fields, field_count, metadata_columns, row.metadata)) {
            ok = 0;
            break;
        }
        strcpy(row.id, id);
        strcpy(row.label, label);
        strcpy(row.photo, photo);
        row.class_id = strcmp(label, "apto") == 0 ? 1 : 0;
        if (!append_row(data, &row)) { ok = 0; break; }
    }
    if (ferror(file)) ok = 0;
    fclose(file);
    return ok && data->count > 0;
}

static study_row *find_row(study_data *data, const char *id) {
    for (size_t i = 0; i < data->count; ++i)
        if (strcmp(data->rows[i].id, id) == 0) return &data->rows[i];
    return NULL;
}

static int load_manifest(const char *path, study_data *data) {
    FILE *file = fopen(path, "r");
    char line[MAX_LINE], *fields[MAX_FIELDS];
    int columns = 0;
    if (!file) return 0;
    if (!read_csv_header(file, line, sizeof(line), fields, MAX_FIELDS, &columns)) {
        fclose(file);
        return 0;
    }
    const int split_col = find_column(fields, columns, "split");
    const int id_col = find_column(fields, columns, "id");
    const int label_col = find_column(fields, columns, "study_label");
    const int photo_col = find_column(fields, columns, "foto_local");
    if (split_col < 0 || id_col < 0 || label_col < 0 || photo_col < 0) {
        fclose(file);
        return 0;
    }
    int ok = 1;
    while (fgets(line, sizeof(line), file)) {
        if (!strchr(line, '\n') && !feof(file)) { ok = 0; break; }
        const int field_count = csv_split(line, fields, MAX_FIELDS);
        const char *split, *id, *label, *photo;
        if (field_count < 0 || !field_at(fields, field_count, split_col, &split) ||
            !field_at(fields, field_count, id_col, &id) ||
            !field_at(fields, field_count, label_col, &label) ||
            !field_at(fields, field_count, photo_col, &photo)) {
            ok = 0;
            break;
        }
        if (strcmp(label, "incierto") == 0) continue;
        study_row *row = find_row(data, id);
        if (!row || row->in_manifest || strcmp(row->label, label) != 0 ||
            strcmp(row->photo, photo) != 0) {
            ok = 0;
            break;
        }
        if (strcmp(split, "train") == 0) row->split = SPLIT_TRAIN;
        else if (strcmp(split, "val") == 0) row->split = SPLIT_VAL;
        else if (strcmp(split, "test") == 0) row->split = SPLIT_TEST;
        else { ok = 0; break; }
        row->in_manifest = 1;
    }
    if (ferror(file)) ok = 0;
    fclose(file);
    for (size_t i = 0; ok && i < data->count; ++i)
        if (!data->rows[i].in_manifest) ok = 0;
    return ok;
}

static void jpeg_fail(j_common_ptr common) {
    jpeg_error_state *error = (jpeg_error_state *)common->err;
    longjmp(error->jump, 1);
}

static int decode_jpeg(const char *path, unsigned char **pixels_out,
                       size_t *width_out, size_t *height_out) {
    FILE *file = fopen(path, "rb");
    jpeg_state *state = (jpeg_state *)calloc(1, sizeof(*state));
    if (!file || !state) {
        if (file) fclose(file);
        free(state);
        return 0;
    }
    state->decoder.err = jpeg_std_error(&state->error.base);
    state->error.base.error_exit = jpeg_fail;
    if (setjmp(state->error.jump)) {
        if (state->created) jpeg_destroy_decompress(&state->decoder);
        free(state->pixels);
        fclose(file);
        free(state);
        return 0;
    }
    jpeg_create_decompress(&state->decoder);
    state->created = 1;
    jpeg_stdio_src(&state->decoder, file);
    if (jpeg_read_header(&state->decoder, TRUE) != JPEG_HEADER_OK) {
        jpeg_destroy_decompress(&state->decoder);
        fclose(file);
        free(state);
        return 0;
    }
    state->decoder.out_color_space = JCS_RGB;
    jpeg_start_decompress(&state->decoder);
    const size_t width = state->decoder.output_width;
    const size_t height = state->decoder.output_height;
    if (state->decoder.output_components != IMAGE_CHANNELS || !width || !height ||
        width > SIZE_MAX / IMAGE_CHANNELS ||
        height > SIZE_MAX / (width * IMAGE_CHANNELS)) {
        jpeg_destroy_decompress(&state->decoder);
        fclose(file);
        free(state);
        return 0;
    }
    state->pixels = (unsigned char *)malloc(width * height * IMAGE_CHANNELS);
    if (!state->pixels) {
        jpeg_destroy_decompress(&state->decoder);
        fclose(file);
        free(state);
        return 0;
    }
    while (state->decoder.output_scanline < state->decoder.output_height) {
        JSAMPROW row = state->pixels +
            (size_t)state->decoder.output_scanline * width * IMAGE_CHANNELS;
        jpeg_read_scanlines(&state->decoder, &row, 1);
    }
    jpeg_finish_decompress(&state->decoder);
    jpeg_destroy_decompress(&state->decoder);
    fclose(file);
    *pixels_out = state->pixels;
    *width_out = width;
    *height_out = height;
    free(state);
    return 1;
}

static double sinc_pi(double value) {
    const double pi = 3.14159265358979323846;
    if (fabs(value) < 1e-12) return 1.0;
    return sin(pi * value) / (pi * value);
}

static double lanczos3(double value) {
    value = fabs(value);
    return value < 3.0 ? sinc_pi(value) * sinc_pi(value / 3.0) : 0.0;
}

static int resize_lanczos_nchw(const unsigned char *source, size_t width,
                               size_t height, float output[IMAGE_CHANNELS *
                                                           IMAGE_SIDE * IMAGE_SIDE]) {
    const size_t intermediate_count = height * IMAGE_SIDE * IMAGE_CHANNELS;
    float *intermediate = (float *)malloc(intermediate_count * sizeof(float));
    if (!intermediate) return 0;
    const double scale_x = (double)width / IMAGE_SIDE;
    const double support_x = 3.0 * fmax(1.0, scale_x);
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < IMAGE_SIDE; ++x) {
            const double center = ((double)x + 0.5) * scale_x - 0.5;
            const long begin = (long)ceil(center - support_x);
            const long end = (long)floor(center + support_x);
            double sums[IMAGE_CHANNELS] = {0.0, 0.0, 0.0};
            double weights = 0.0;
            for (long sx = begin; sx <= end; ++sx) {
                const size_t clamped = sx < 0 ? 0 :
                    (size_t)sx >= width ? width - 1 : (size_t)sx;
                const double weight = lanczos3((sx - center) / fmax(1.0, scale_x));
                weights += weight;
                for (size_t c = 0; c < IMAGE_CHANNELS; ++c)
                    sums[c] += weight * source[(y * width + clamped) *
                                               IMAGE_CHANNELS + c];
            }
            for (size_t c = 0; c < IMAGE_CHANNELS; ++c)
                intermediate[(y * IMAGE_SIDE + x) * IMAGE_CHANNELS + c] =
                    (float)(sums[c] / weights);
        }
    }
    const double scale_y = (double)height / IMAGE_SIDE;
    const double support_y = 3.0 * fmax(1.0, scale_y);
    for (size_t y = 0; y < IMAGE_SIDE; ++y) {
        const double center = ((double)y + 0.5) * scale_y - 0.5;
        const long begin = (long)ceil(center - support_y);
        const long end = (long)floor(center + support_y);
        for (size_t x = 0; x < IMAGE_SIDE; ++x) {
            double sums[IMAGE_CHANNELS] = {0.0, 0.0, 0.0};
            double weights = 0.0;
            for (long sy = begin; sy <= end; ++sy) {
                const size_t clamped = sy < 0 ? 0 :
                    (size_t)sy >= height ? height - 1 : (size_t)sy;
                const double weight = lanczos3((sy - center) / fmax(1.0, scale_y));
                weights += weight;
                for (size_t c = 0; c < IMAGE_CHANNELS; ++c)
                    sums[c] += weight * intermediate[(clamped * IMAGE_SIDE + x) *
                                                      IMAGE_CHANNELS + c];
            }
            for (size_t c = 0; c < IMAGE_CHANNELS; ++c) {
                const float value = (float)(sums[c] / weights) / 255.0f;
                output[c * IMAGE_SIDE * IMAGE_SIDE + y * IMAGE_SIDE + x] =
                    fminf(1.0f, fmaxf(0.0f, value));
            }
        }
    }
    free(intermediate);
    return 1;
}

static int encoder_create(image_encoder *encoder, const char *checkpoint) {
    memset(encoder, 0, sizeof(*encoder));
    const tc_conv2d_options options = {1, 1, 1, 1};
    encoder->conv1 = tc_conv2d_layer_create(3, 8, 3, 3, options, 101u);
    encoder->conv2 = tc_conv2d_layer_create(8, 16, 3, 3, options, 202u);
    encoder->projection = tc_dense_layer_create(1024, VISUAL_FEATURES, 303u);
    if (!encoder->conv1 || !encoder->conv2 || !encoder->projection) return 0;

    size_t count = 0;
    tc_tensor_nd **saved = tc_load_checkpoint(checkpoint, &count, 0);
    if (!saved || count != 10) {
        tc_free_checkpoint(saved, count);
        return 0;
    }
    tc_tensor_nd *parameters[] = {
        tc_conv2d_layer_kernel(encoder->conv1),
        tc_conv2d_layer_bias(encoder->conv1),
        tc_conv2d_layer_kernel(encoder->conv2),
        tc_conv2d_layer_bias(encoder->conv2),
        tc_dense_layer_weight(encoder->projection),
        tc_dense_layer_bias(encoder->projection),
    };
    int ok = 1;
    for (size_t i = 0; i < 6; ++i)
        if (!tc_tensor_nd_copy_(parameters[i], saved[i])) ok = 0;
    tc_free_checkpoint(saved, count);
    return ok;
}

static void encoder_free(image_encoder *encoder) {
    tc_conv2d_layer_free(encoder->conv1);
    tc_conv2d_layer_free(encoder->conv2);
    tc_dense_layer_free(encoder->projection);
    memset(encoder, 0, sizeof(*encoder));
}

static tc_tensor_nd *relu_free(tc_tensor_nd *input) {
    tc_tensor_nd *output = input ? tc_tensor_nd_relu(input) : NULL;
    tc_tensor_nd_free(input);
    return output;
}

static int encoder_extract(image_encoder *encoder,
                           const float image[IMAGE_CHANNELS * IMAGE_SIDE *
                                              IMAGE_SIDE],
                           float output[VISUAL_FEATURES]) {
    const size_t input_shape[] = {1, IMAGE_CHANNELS, IMAGE_SIDE, IMAGE_SIDE};
    const size_t flat_shape[] = {1, 1024};
    tc_tensor_nd *input = tc_tensor_nd_from_data(input_shape, 4, image, 0);
    tc_tensor_nd *value = input
        ? tc_conv2d_layer_forward(encoder->conv1, input) : NULL;
    tc_tensor_nd_free(input);
    value = relu_free(value);
    tc_tensor_nd *pooled = value ? tc_tensor_nd_maxpool2d(value, 2, 2) : NULL;
    tc_tensor_nd_free(value);
    value = pooled ? tc_conv2d_layer_forward(encoder->conv2, pooled) : NULL;
    tc_tensor_nd_free(pooled);
    value = relu_free(value);
    pooled = value ? tc_tensor_nd_maxpool2d(value, 2, 2) : NULL;
    tc_tensor_nd_free(value);
    value = pooled ? tc_tensor_nd_reshape_to(pooled, flat_shape, 2) : NULL;
    tc_tensor_nd_free(pooled);
    tc_tensor_nd *embedding = value
        ? tc_dense_layer_forward(encoder->projection, value) : NULL;
    tc_tensor_nd_free(value);
    embedding = relu_free(embedding);
    const float *features = tc_tensor_nd_const_data(embedding);
    int ok = features != NULL;
    for (size_t i = 0; ok && i < VISUAL_FEATURES; ++i)
        if (!isfinite(features[i])) ok = 0;
    if (ok) memcpy(output, features, VISUAL_FEATURES * sizeof(float));
    tc_tensor_nd_free(embedding);
    return ok;
}

static int join_path(char *output, size_t capacity, const char *root,
                     const char *relative) {
    if (!root || !relative || relative[0] == '/' || strstr(relative, ".."))
        return 0;
    const int written = snprintf(output, capacity, "%s/%s", root, relative);
    return written > 0 && (size_t)written < capacity;
}

static int encode_photo_path(image_encoder *encoder, const char *path,
                             float visual[VISUAL_FEATURES]) {
    unsigned char *pixels = NULL;
    size_t width = 0, height = 0;
    float image[IMAGE_CHANNELS * IMAGE_SIDE * IMAGE_SIDE];
    const int ok = decode_jpeg(path, &pixels, &width, &height) &&
                   resize_lanczos_nchw(pixels, width, height, image) &&
                   encoder_extract(encoder, image, visual);
    free(pixels);
    return ok;
}

static int prepare_features(study_data *data, const char *data_dir,
                            image_encoder *encoder, double *elapsed_out) {
    const double started = seconds_now();
    size_t processed = 0;
    for (size_t i = 0; i < data->count; ++i) {
        study_row *row = &data->rows[i];
        if (row->split != SPLIT_TRAIN && row->split != SPLIT_TEST) continue;
        char path[MAX_PATH * 2];
        if (!join_path(path, sizeof(path), data_dir, row->photo)) return 0;
        if (!encode_photo_path(encoder, path, row->observation)) return 0;
        memcpy(row->observation + VISUAL_FEATURES, row->metadata,
               METADATA_FEATURES * sizeof(float));
        ++processed;
    }
    if (!processed) return 0;
    *elapsed_out = seconds_now() - started;
    return 1;
}

static int bandit_reset(void *user_data, uint32_t seed, float *observation_out) {
    bandit_state *state = (bandit_state *)user_data;
    if (!state || !state->train_count || !observation_out) return 0;
    state->rng = seed ? seed : 1u;
    state->current = state->train_rows[
        random_u32(&state->rng) % state->train_count];
    memcpy(observation_out, state->current->observation,
           OBSERVATION_SIZE * sizeof(float));
    return 1;
}

static int bandit_step(void *user_data, size_t action, float *observation_out,
                       float *reward_out, int *terminated_out,
                       int *truncated_out) {
    bandit_state *state = (bandit_state *)user_data;
    if (!state || !state->current || action >= ACTIONS || !observation_out ||
        !reward_out || !terminated_out || !truncated_out)
        return 0;
    memcpy(observation_out, state->current->observation,
           OBSERVATION_SIZE * sizeof(float));
    *reward_out = action == (size_t)state->current->class_id ? 1.0f : -1.0f;
    *terminated_out = 1;
    *truncated_out = 0;
    return 1;
}

static int make_path(char *output, size_t capacity, const char *root,
                     const char *name) {
    const int written = snprintf(output, capacity, "%s/%s", root, name);
    return written > 0 && (size_t)written < capacity;
}

static int evaluate(tc_dqn_agent *agent, study_data *data) {
    size_t matrix[2][2] = {{0, 0}, {0, 0}};
    size_t total = 0, correct = 0;
    for (size_t i = 0; i < data->count; ++i) {
        study_row *row = &data->rows[i];
        if (row->split != SPLIT_TEST) continue;
        size_t action = 0;
        if (!tc_dqn_select_action(agent, row->observation, 0.0f, &action))
            return 0;
        ++matrix[row->class_id][action];
        correct += action == (size_t)row->class_id;
        ++total;
    }
    if (!total) return 0;
    const size_t tn = matrix[0][0], fp = matrix[0][1];
    const size_t fn = matrix[1][0], tp = matrix[1][1];
    const double precision = tp + fp ? (double)tp / (tp + fp) : 0.0;
    const double recall = tp + fn ? (double)tp / (tp + fn) : 0.0;
    const double f1 = precision + recall
        ? 2.0 * precision * recall / (precision + recall) : 0.0;
    printf("prueba: %zu | exactitud %.2f%% | precision apto %.2f%% | "
           "recall apto %.2f%% | F1 apto %.2f%% | matriz [[%zu,%zu],[%zu,%zu]]\n",
           total, 100.0 * (double)correct / total, 100.0 * precision,
           100.0 * recall, 100.0 * f1, tn, fp, fn, tp);
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 4 && argc != 12) {
        fprintf(stderr,
                "Uso: %s DATASET_DIR MANIFEST.csv CNN_CHECKPOINT.tck "
                "[PHOTO TEMP_C HUMIDITY_PCT PEOPLE HAS_TABLE OCCUPANCY "
                "SPACE_TYPE SOUND_DBFS]\n", argv[0]);
        return 2;
    }
    char csv_path[MAX_PATH * 2];
    if (!make_path(csv_path, sizeof(csv_path), argv[1], "datos.csv")) {
        fprintf(stderr, "Ruta del dataset demasiado larga.\n");
        return 2;
    }
    study_data data = {0};
    image_encoder encoder = {0};
    tc_dqn_agent *agent = NULL;
    study_row **train_rows = NULL;
    int result = 1;

    if (!load_data_csv(csv_path, &data) ||
        !load_manifest(argv[2], &data)) {
        fprintf(stderr, "No se pudieron validar datos.csv y manifest.csv.\n");
        goto cleanup;
    }
    size_t train_count = 0, test_count = 0;
    for (size_t i = 0; i < data.count; ++i) {
        train_count += data.rows[i].split == SPLIT_TRAIN;
        test_count += data.rows[i].split == SPLIT_TEST;
    }
    if (!train_count || !test_count) {
        fprintf(stderr, "El manifest debe incluir train y test no vacíos.\n");
        goto cleanup;
    }
    if (!encoder_create(&encoder, argv[3])) {
        fprintf(stderr, "No se pudo cargar el CNN (se esperan 10 tensores compatibles).\n");
        goto cleanup;
    }
    double feature_seconds = 0.0;
    if (!prepare_features(&data, argv[1], &encoder, &feature_seconds)) {
        fprintf(stderr, "Falló la extracción visual; revisa fotos y checkpoint.\n");
        goto cleanup;
    }

    train_rows = (study_row **)malloc(train_count * sizeof(*train_rows));
    if (!train_rows) goto cleanup;
    size_t next_train = 0;
    for (size_t i = 0; i < data.count; ++i)
        if (data.rows[i].split == SPLIT_TRAIN)
            train_rows[next_train++] = &data.rows[i];
    bandit_state state = {
        .train_rows = train_rows,
        .train_count = train_count,
        .rng = 42u,
    };
    tc_env env = {
        .observation_size = OBSERVATION_SIZE,
        .action_count = ACTIONS,
        .user_data = &state,
        .reset = bandit_reset,
        .step = bandit_step,
    };
    const tc_dqn_config config = {
        .observation_size = OBSERVATION_SIZE,
        .action_count = ACTIONS,
        .hidden_size = 64,
        .replay_capacity = 1024,
        .batch_size = 32,
        .target_update_interval = 100,
        .learning_rate = 0.0005f,
        .discount_factor = 0.0f,
        .seed = 42u,
    };
    agent = tc_dqn_agent_create(&config);
    if (!agent) {
        fprintf(stderr, "No se pudo crear el DQN.\n");
        tc_env_close(&env);
        goto cleanup;
    }

    const double train_started = seconds_now();
    size_t correct_training_actions = 0;
    for (size_t episode = 0; episode < EPISODES; ++episode) {
        float observation[OBSERVATION_SIZE], next_observation[OBSERVATION_SIZE];
        const uint32_t seed = (uint32_t)(42u + episode);
        if (!tc_env_reset(&env, seed, observation)) {
            fprintf(stderr, "Falló reset del contextual bandit.\n");
            tc_env_close(&env);
            goto cleanup;
        }
        const float progress = (float)episode / (float)(EPISODES - 1);
        const float epsilon = 1.0f - 0.95f * progress;
        size_t action = 0;
        if (!tc_dqn_select_action(agent, observation, epsilon, &action)) {
            fprintf(stderr, "Falló la selección de acción.\n");
            tc_env_close(&env);
            goto cleanup;
        }
        float reward = 0.0f;
        int terminated = 0, truncated = 0;
        if (!tc_env_step(&env, action, next_observation, &reward,
                         &terminated, &truncated) ||
            !tc_dqn_remember(agent, observation, action, reward,
                             next_observation, terminated)) {
            fprintf(stderr, "Falló el paso o almacenamiento de transición.\n");
            tc_env_close(&env);
            goto cleanup;
        }
        correct_training_actions += reward > 0.0f;
        float loss = 0.0f;
        if (tc_dqn_train_step(agent, &loss) == TC_DQN_TRAIN_ERROR) {
            fprintf(stderr, "Falló el entrenamiento DQN.\n");
            tc_env_close(&env);
            goto cleanup;
        }
    }
    tc_env_close(&env);
    printf("contextos: %zu train, %zu test, %zu inciertos excluidos\n",
           train_count, test_count, data.uncertain_count);
    printf("CNN congelado: 16 características visuales + 11 metadatos = %d\n",
           OBSERVATION_SIZE);
    printf("extracción de características C: %.3f s\n", feature_seconds);
    printf("DQN contextual bandit: %d episodios | %.3f s | "
           "aciertos explorando %.2f%%\n", EPISODES,
           seconds_now() - train_started,
           100.0 * (double)correct_training_actions / EPISODES);
    if (!evaluate(agent, &data)) {
        fprintf(stderr, "No se pudo evaluar el split test.\n");
        goto cleanup;
    }
    if (argc == 12) {
        const int columns[] = {0, 1, 2, 3, 4, 5, 6};
        char *metadata_fields[] = {
            argv[7], argv[8], argv[9], argv[10], argv[5], argv[6], argv[11],
        };
        float metadata[METADATA_FEATURES];
        float visual[VISUAL_FEATURES];
        float observation[OBSERVATION_SIZE];
        size_t action = 0;
        if (!encode_metadata(metadata_fields, 7, columns, metadata) ||
            !encode_photo_path(&encoder, argv[4], visual)) {
            fprintf(stderr, "No se pudieron procesar la foto o sus metadatos.\n");
            goto cleanup;
        }
        memcpy(observation, visual, sizeof(visual));
        memcpy(observation + VISUAL_FEATURES, metadata, sizeof(metadata));
        if (!tc_dqn_select_action(agent, observation, 0.0f, &action)) {
            fprintf(stderr, "El DQN no pudo clasificar la foto.\n");
            goto cleanup;
        }
        printf("foto nueva (DQN): %s\n", action ? "apto" : "no_apto");
    }
    result = 0;

cleanup:
    tc_dqn_agent_free(agent);
    free(train_rows);
    encoder_free(&encoder);
    free(data.rows);
    return result;
}
