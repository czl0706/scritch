#include "scritch_nn.h"

// static void linear_init(LinearLayer *layer, int input_size, int output_size, float *weight, float *bias) {
//     layer->weight = weight;
//     layer->bias = bias;
//     layer->input_size = input_size;
//     layer->output_size = output_size;
// }

static void relu_inplace(float *input, int size) {
    for (int i = 0; i < size; ++i) {
        if (input[i] < 0) { input[i] = 0; }
    }
} 

// LinearLayer net1_1, net1_2, net1_3, net2_1, net2_3;

float *net1_output; 
float *net2_1_output; 
float *net2_3_output;

void scritch_init() {
    // linear_init(&net1_1, 45, 20, (float *)net1_1_0_weight, (float *)net1_1_0_bias);
    // linear_init(&net1_2, 20, 20, (float *)net1_2_0_weight, (float *)net1_2_0_bias);
    // linear_init(&net1_3, 20, 20, (float *)net1_3_0_weight, (float *)net1_3_0_bias);
    // linear_init(&net2_1, 20, 40, (float *)net2_1_weight, (float *)net2_1_bias);
    // linear_init(&net2_3, 40, 2, (float *)net2_3_weight, (float *)net2_3_bias);

    net1_output = (float *)malloc(sizeof(float) * 20 * 3);
    net2_1_output = (float *)malloc(sizeof(float) * 40);
    net2_3_output = (float *)malloc(sizeof(float) * 2);
}


void scritch_forward(float *accX, float *accY, float *accZ, float *output) {
    ESP_ERROR_CHECK(dspm_mult_f32_ae32(net1_1_weight, (const float *)accX, net1_output, 20, 45, 1));
    ESP_ERROR_CHECK(dsps_add_f32_ae32((const float *)net1_output, net1_1_bias, net1_output, 20, 1, 1, 1));
    relu_inplace(net1_output, 20);

    ESP_ERROR_CHECK(dspm_mult_f32_ae32(net1_2_weight, (const float *)accY, (net1_output + 20), 20, 45, 1));
    ESP_ERROR_CHECK(dsps_add_f32_ae32((const float *)(net1_output + 20), net1_2_bias, (net1_output + 20), 20, 1, 1, 1));
    relu_inplace((net1_output + 20), 20);

    ESP_ERROR_CHECK(dspm_mult_f32_ae32(net1_3_weight, (const float *)accZ, (net1_output + 40), 20, 45, 1));
    ESP_ERROR_CHECK(dsps_add_f32_ae32((const float *)(net1_output + 40), net1_3_bias, (net1_output + 40), 20, 1, 1, 1));
    relu_inplace((net1_output + 40), 20);

    ESP_ERROR_CHECK(dspm_mult_f32_ae32(net2_1_weight, (const float *)net1_output, net2_1_output, 40, 60, 1));
    ESP_ERROR_CHECK(dsps_add_f32_ae32((const float *)net2_1_output, net2_1_bias, net2_1_output, 40, 1, 1, 1));
    relu_inplace(net2_1_output, 40);

    ESP_ERROR_CHECK(dspm_mult_f32_ae32(net2_3_weight, (const float *)net2_1_output, net2_3_output, 2, 40, 1));
    ESP_ERROR_CHECK(dsps_add_f32_ae32((const float *)net2_3_output, net2_3_bias, net2_3_output, 2, 1, 1, 1));

    output[0] = net2_3_output[0];
    output[1] = net2_3_output[1];
}
