#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include "../weights/weights.h"
#include "../weights/validation_data.h"

/* ---------------------------------------------------------------------
 * Buffers
 *
 * Activations are stored as float between layers. This matches the
 * Python reference exactly: after each matmul we dequantize to float,
 * add the float bias, apply ReLU, then re-quantize to int8 with a
 * FRESH scale/zero-point computed from that activation's own min/max
 * (same as quantize_array() in week4_train.py).
 * ------------------------------------------------------------------- */
static int8_t x_q[INPUT_SIZE];
static int8_t h1_q[HIDDEN1_SIZE];
static int8_t h2_q[HIDDEN2_SIZE];

static float h1_f[HIDDEN1_SIZE];
static float h2_f[HIDDEN2_SIZE];
static float out_f[OUTPUT_SIZE];

/* ---------------------------------------------------------------------
 * quantize_array
 *
 * Direct C port of the Python quantize_array() function. Computes a
 * fresh scale + zero_point from the array's own min/max, exactly the
 * way week4_train.py does for activations between layers.
 *
 * out_q       : destination int8 buffer
 * out_scale   : scale factor used (written back, needed by next layer)
 * out_zp      : zero point used (written back, needed by next layer)
 * ------------------------------------------------------------------- */
static void quantize_array(const float *in, int len,
                            int8_t *out_q, float *out_scale, int *out_zp) {
    float vmin = in[0], vmax = in[0];
    for (int i = 1; i < len; i++) {
        if (in[i] < vmin) vmin = in[i];
        if (in[i] > vmax) vmax = in[i];
    }

    if (vmax == vmin) {
        for (int i = 0; i < len; i++) out_q[i] = 0;
        *out_scale = 1.0f;
        *out_zp = 0;
        return;
    }

    float scale = (vmax - vmin) / 255.0f;   /* q_max - q_min = 127 - (-128) = 255 */
    int zp = (int)lroundf(-128.0f - vmin / scale);
    if (zp < -128) zp = -128;
    if (zp >  127) zp =  127;

    for (int i = 0; i < len; i++) {
        int32_t q = (int32_t)lroundf(in[i] / scale + (float)zp);
        if (q < -128) q = -128;
        if (q >  127) q =  127;
        out_q[i] = (int8_t)q;
    }

    *out_scale = scale;
    *out_zp = zp;
}

/* ---------------------------------------------------------------------
 * quantized_dense_layer
 *
 * Direct C port of the int8 path inside week4_train.py's validation
 * loop:
 *
 *   acc   = (W_q - W_zp) @ (x_q - x_zp)          [int32 accumulator]
 *   out_f = acc * x_scale * W_scale + bias        [float rescale + bias]
 *
 * This is the part the original skeleton got wrong: it added bias
 * directly into the int32 accumulator and never subtracted zero
 * points. Both are required for the math to match Python.
 * ------------------------------------------------------------------- */
static void quantized_dense_layer(const int8_t *W, const float *bias,
                                   const int8_t *x,
                                   float x_scale, int x_zp,
                                   float W_scale, int W_zp,
                                   float *out_f,
                                   int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        int32_t acc = 0;
        for (int j = 0; j < cols; j++) {
            int32_t w = (int32_t)W[i * cols + j] - W_zp;
            int32_t xv = (int32_t)x[j] - x_zp;
            acc += w * xv;
        }
        out_f[i] = (float)acc * x_scale * W_scale + bias[i];
    }
}

static void relu(float *buf, int len) {
    for (int i = 0; i < len; i++)
        if (buf[i] < 0.0f) buf[i] = 0.0f;
}

static int argmax_f(const float *buf, int len) {
    int best = 0;
    for (int i = 1; i < len; i++)
        if (buf[i] > buf[best]) best = i;
    return best;
}

/* ---------------------------------------------------------------------
 * normalize_input
 *
 * Applies (x - mean) / std using NORM_MEAN / NORM_STD from weights.h,
 * matching the normalization step in week4_train.py section 1.
 * ------------------------------------------------------------------- */
static void normalize_input(const float *raw, float *out, int len) {
    for (int i = 0; i < len; i++)
        out[i] = (raw[i] - NORM_MEAN[i]) / NORM_STD[i];
}

/* ---------------------------------------------------------------------
 * infer
 *
 * raw_features : 13 raw MFCC coefficients (NOT yet normalized)
 * returns      : predicted class index (0=yes, 1=no, 2=stop)
 * ------------------------------------------------------------------- */
int infer(const float *raw_features) {
    float normed[INPUT_SIZE];
    normalize_input(raw_features, normed, INPUT_SIZE);

    /* Quantize input -- fresh scale/zero-point, same as Python x_q, x_s, x_zp */
    float x_scale;
    int x_zp;
    quantize_array(normed, INPUT_SIZE, x_q, &x_scale, &x_zp);

    /* Layer 1: 13 -> 64 */
    quantized_dense_layer(W1_data, b1_data, x_q,
                           x_scale, x_zp, SCALE_W1, ZP_W1,
                           h1_f, HIDDEN1_SIZE, INPUT_SIZE);
    relu(h1_f, HIDDEN1_SIZE);
    float h1_scale;
    int h1_zp;
    quantize_array(h1_f, HIDDEN1_SIZE, h1_q, &h1_scale, &h1_zp);

    /* Layer 2: 64 -> 32 */
    quantized_dense_layer(W2_data, b2_data, h1_q,
                           h1_scale, h1_zp, SCALE_W2, ZP_W2,
                           h2_f, HIDDEN2_SIZE, HIDDEN1_SIZE);
    relu(h2_f, HIDDEN2_SIZE);
    float h2_scale;
    int h2_zp;
    quantize_array(h2_f, HIDDEN2_SIZE, h2_q, &h2_scale, &h2_zp);

    /* Layer 3: 32 -> 3 (no ReLU, no re-quantization -- raw logits) */
    quantized_dense_layer(W3_data, b3_data, h2_q,
                           h2_scale, h2_zp, SCALE_W3, ZP_W3,
                           out_f, OUTPUT_SIZE, HIDDEN2_SIZE);

    return argmax_f(out_f, OUTPUT_SIZE);
}

static void memory_report(void) {
    size_t weights = sizeof(W1_data) + sizeof(W2_data) + sizeof(W3_data);
    size_t biases  = sizeof(b1_data) + sizeof(b2_data) + sizeof(b3_data);
    size_t activations = sizeof(x_q) + sizeof(h1_q) + sizeof(h2_q)
                        + sizeof(h1_f) + sizeof(h2_f) + sizeof(out_f);
    size_t total = weights + biases + activations;

    printf("Weights     : %4zu bytes\n", weights);
    printf("Biases      : %4zu bytes\n", biases);
    printf("Activations : %4zu bytes\n", activations);
    printf("Total       : %4zu bytes (%.2f KB)\n", total, total / 1024.0);
    printf("Budget      : 262144 bytes (256 KB)\n");
    printf("Remaining   : %4zu bytes (%.2f KB)\n",
           262144 - total, (262144 - total) / 1024.0);
}

/* ---------------------------------------------------------------------
 * main
 *
 * Runs inference over the REAL validation set (exported from
 * data/X_val.npy / data/y_val.npy by python/export_validation_data.py)
 * and reports accuracy + average inference time. This is the actual
 * Week 5 deliverable: proving the C engine reproduces Python's int8
 * validation accuracy, not just that it runs.
 * ------------------------------------------------------------------- */
int main(void) {
    printf("=== TinyML Inference Engine ===\n\n");
    memory_report();

    const char *labels[OUTPUT_SIZE] = {"yes", "no", "stop"};

    printf("\nRunning inference on %d real validation samples...\n", NUM_VAL_SAMPLES);

    int correct = 0;
    clock_t t0 = clock();
    for (int i = 0; i < NUM_VAL_SAMPLES; i++) {
        int pred = infer(VAL_FEATURES[i]);
        if (pred == VAL_LABELS[i]) correct++;
    }
    clock_t t1 = clock();

    double total_ms = 1000.0 * (t1 - t0) / CLOCKS_PER_SEC;
    double ms_per_inference = total_ms / NUM_VAL_SAMPLES;
    double accuracy = 100.0 * correct / NUM_VAL_SAMPLES;

    printf("\nC engine int8 accuracy : %d / %d  (%.1f%%)\n",
           correct, NUM_VAL_SAMPLES, accuracy);
    printf("Compare to Python int8 accuracy reported by week4_train.py.\n");
    printf("Average inference time : %.4f ms\n", ms_per_inference);

    /* A few example predictions for sanity checking by eye */
    printf("\nSample predictions:\n");
    int n_show = NUM_VAL_SAMPLES < 5 ? NUM_VAL_SAMPLES : 5;
    for (int i = 0; i < n_show; i++) {
        int pred = infer(VAL_FEATURES[i]);
        printf("  sample[%d] -> predicted: %-5s actual: %-5s %s\n",
               i, labels[pred], labels[VAL_LABELS[i]],
               pred == VAL_LABELS[i] ? "OK" : "MISS");
    }

    return 0;
}