#include "scritch_nn.h"
#include "model_weights.h"

static inline void conv1d(const float* input, const float* weight, const float* bias, float* output,
                   int in_channels, int out_channels, int input_size, int kernel_size, int stride, int padding) {
    for (int o = 0; o < out_channels; ++o) {
        for (int oh = 0; oh < input_size; oh += stride) {
            int hstart = oh - padding;
            int hend = hstart + kernel_size;

            // Apply padding
            hstart = (hstart < 0) ? 0 : hstart;
            hend = (hend > input_size) ? input_size : hend;

            float result = 0.0;
            for (int i = 0; i < in_channels; ++i) {
                for (int k = 0; k < kernel_size; ++k) {
                    int idx = hstart + k;
                    if (idx >= 0 && idx < input_size) {
                        result += input[i * input_size + idx] * weight[o * in_channels * kernel_size + i * kernel_size + k];
                    }
                }
            }
            result += bias[o];
            output[o * (input_size / stride) + oh / stride] = result;  // Output size is determined by input size, stride, and padding
        }
    }
}

static inline void relu_inplace(float *input, int size) {
    for (int i = 0; i < size; ++i) {
        if (input[i] < 0) { input[i] = 0; }
    }
} 

static float *input_data;
static float *conv1_output;
static float *fc1_output;
static float *fc2_output;

bool scritch_forward() {
    conv1d(input_data, (const float*)conv1_weight, conv1_bias, conv1_output, 3, 6, 50, 5, 2, 2);
    relu_inplace(conv1_output, 6 * (50 / 2));

    dspm_mult_f32_ansi(net1_weight, (const float *)conv1_output, fc1_output, 30, 150, 1);
    dsps_add_f32_ansi((const float *)fc1_output, net1_bias, fc1_output, 30, 1, 1, 1);
    relu_inplace(fc1_output, 30);

    dspm_mult_f32_ansi(net2_weight, (const float *)fc1_output, fc2_output, 2, 30, 1);
    dsps_add_f32_ansi((const float *)fc2_output, net2_bias, fc2_output, 2, 1, 1, 1);

    return fc2_output[1] > fc2_output[0];
}

void scritch_init(float *_input_data) {
    input_data = _input_data;
    conv1_output = (float *)malloc(sizeof(float) * 150);
    fc1_output   = (float *)malloc(sizeof(float) * 30);
    fc2_output   = (float *)malloc(sizeof(float) * 2);
}

void scritch_deinit() {
    free(conv1_output);
    free(fc1_output);
    free(fc2_output);
}