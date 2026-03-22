#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define INPUT_SIZE    13
#define HIDDEN1_SIZE  64
#define HIDDEN2_SIZE  32
#define OUTPUT_SIZE    3

static int8_t  W1[HIDDEN1_SIZE][INPUT_SIZE];
static int8_t  W2[HIDDEN2_SIZE][HIDDEN1_SIZE];
static int8_t  W3[OUTPUT_SIZE][HIDDEN2_SIZE];
static int32_t b1[HIDDEN1_SIZE];
static int32_t b2[HIDDEN2_SIZE];
static int32_t b3[OUTPUT_SIZE];

static int32_t act1[HIDDEN1_SIZE];
static int32_t act2[HIDDEN2_SIZE];
static int32_t act3[OUTPUT_SIZE];


static void relu(int32_t *buf, int len) {
    for (int i = 0; i < len; i++)
        if (buf[i] < 0) buf[i] = 0;
}

static void matmul(const int8_t *W, const int8_t *x,
                   const int32_t *bias, int32_t *out,
                   int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        int32_t acc = bias[i];
        for (int j = 0; j < cols; j++)
            acc += (int32_t)W[i * cols + j] * (int32_t)x[j];
        out[i] = acc;
    }
}

static int argmax(const int32_t *buf, int len) {
    int best = 0;
    for (int i = 1; i < len; i++)
        if (buf[i] > buf[best]) best = i;
    return best;
}

static void quantize_input(const float *in, int8_t *out, int len) {
    float scale = 0.1f;
    for (int i = 0; i < len; i++) {
        int32_t q = (int32_t)(in[i] / scale);
        if (q < -128) q = -128;
        if (q >  127) q =  127;
        out[i] = (int8_t)q;
    }
}

static void requantize(const int32_t *in, int8_t *out, int len) {
    for (int i = 0; i < len; i++) {
        int32_t q = in[i] / 256;
        if (q < -128) q = -128;
        if (q >  127) q =  127;
        out[i] = (int8_t)q;
    }
}

int infer(const float *features) {
    int8_t x[INPUT_SIZE];
    int8_t h1[HIDDEN1_SIZE];
    int8_t h2[HIDDEN2_SIZE];

    quantize_input(features, x, INPUT_SIZE);

    matmul((int8_t*)W1, x,  b1, act1, HIDDEN1_SIZE, INPUT_SIZE);
    relu(act1, HIDDEN1_SIZE);
    requantize(act1, h1, HIDDEN1_SIZE);

    matmul((int8_t*)W2, h1, b2, act2, HIDDEN2_SIZE, HIDDEN1_SIZE);
    relu(act2, HIDDEN2_SIZE);
    requantize(act2, h2, HIDDEN2_SIZE);

    matmul((int8_t*)W3, h2, b3, act3, OUTPUT_SIZE,  HIDDEN2_SIZE);

    return argmax(act3, OUTPUT_SIZE);
}

void memory_report() {
    size_t weights    = sizeof(W1) + sizeof(W2) + sizeof(W3);
    size_t biases     = sizeof(b1) + sizeof(b2) + sizeof(b3);
    size_t activations= sizeof(act1) + sizeof(act2) + sizeof(act3);
    size_t total      = weights + biases + activations;

    printf("Weights     : %4zu bytes\n", weights);
    printf("Biases      : %4zu bytes\n", biases);
    printf("Activations : %4zu bytes\n", activations);
    printf("Total       : %4zu bytes (%.2f KB)\n", total, total/1024.0);
    printf("Budget      : 262144 bytes (256 KB)\n");
    printf("Remaining   : %4zu bytes (%.2f KB)\n",
           262144 - total, (262144 - total) / 1024.0);
}

int main() {
    for (int i = 0; i < HIDDEN1_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            W1[i][j] = (int8_t)((i * INPUT_SIZE + j) % 127);

    for (int i = 0; i < HIDDEN2_SIZE; i++)
        for (int j = 0; j < HIDDEN1_SIZE; j++)
            W2[i][j] = (int8_t)((i + j) % 127 - 63);

    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN2_SIZE; j++)
            W3[i][j] = (int8_t)((i * 17 + j) % 127 - 63);

    for (int i = 0; i < HIDDEN1_SIZE; i++) b1[i] = i;
    for (int i = 0; i < HIDDEN2_SIZE; i++) b2[i] = 0;
    for (int i = 0; i < OUTPUT_SIZE;  i++) b3[i] = i * 100;

    printf("=== TinyML Inference Engine ===\n\n");
    memory_report();

    const char *labels[OUTPUT_SIZE] = {"yes", "no", "stop"};
    float test[3][INPUT_SIZE] = {
        { 0.1f,  0.3f,  0.5f,  0.2f, -0.1f,  0.4f,  0.3f,
         -0.2f,  0.1f,  0.5f,  0.2f, -0.3f,  0.1f},
        {-0.2f, -0.4f,  0.1f, -0.3f,  0.5f, -0.1f,  0.2f,
          0.4f, -0.3f,  0.1f, -0.5f,  0.3f, -0.1f},
        { 0.5f, -0.1f,  0.3f,  0.1f,  0.2f, -0.4f,  0.1f,
          0.3f, -0.5f,  0.2f,  0.4f, -0.1f,  0.3f},
    };

    printf("\nResults (dummy weights)\n");
    clock_t t0 = clock();
    for (int i = 0; i < 3; i++)
        printf("  input[%d] -> %s\n", i, labels[infer(test[i])]);
    double ms = 1000.0 * (clock() - t0) / CLOCKS_PER_SEC / 3.0;
    printf("\n%.4f ms per inference\n", ms);

    return 0;
}