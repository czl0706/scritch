#include "scritch_nn.h"

static inline void relu_inplace(float *input, int size) {
    for (int i = 0; i < size; ++i) {
        if (input[i] < 0) { input[i] = 0; }
    }
} 

float *net1_output; 
float *net2_1_output; 
float *net2_3_output;

void scritch_init() {
    net1_output = (float *)malloc(sizeof(float) * 20 * 3);
    net2_1_output = (float *)malloc(sizeof(float) * 40);
    net2_3_output = (float *)malloc(sizeof(float) * 2);
}

// void scritch_forward(float *accX, float *accY, float *accZ, float *output) {
//     ESP_ERROR_CHECK(dspm_mult_f32_ae32(net1_1_weight, (const float *)accX, net1_output, 20, 45, 1));
//     ESP_ERROR_CHECK(dsps_add_f32_ae32((const float *)net1_output, net1_1_bias, net1_output, 20, 1, 1, 1));
//     relu_inplace(net1_output, 20);

//     ESP_ERROR_CHECK(dspm_mult_f32_ae32(net1_2_weight, (const float *)accY, (net1_output + 20), 20, 45, 1));
//     ESP_ERROR_CHECK(dsps_add_f32_ae32((const float *)(net1_output + 20), net1_2_bias, (net1_output + 20), 20, 1, 1, 1));
//     relu_inplace((net1_output + 20), 20);

//     ESP_ERROR_CHECK(dspm_mult_f32_ae32(net1_3_weight, (const float *)accZ, (net1_output + 40), 20, 45, 1));
//     ESP_ERROR_CHECK(dsps_add_f32_ae32((const float *)(net1_output + 40), net1_3_bias, (net1_output + 40), 20, 1, 1, 1));
//     relu_inplace((net1_output + 40), 20);

//     ESP_ERROR_CHECK(dspm_mult_f32_ae32(net2_1_weight, (const float *)net1_output, net2_1_output, 40, 60, 1));
//     ESP_ERROR_CHECK(dsps_add_f32_ae32((const float *)net2_1_output, net2_1_bias, net2_1_output, 40, 1, 1, 1));
//     relu_inplace(net2_1_output, 40);

//     ESP_ERROR_CHECK(dspm_mult_f32_ae32(net2_3_weight, (const float *)net2_1_output, net2_3_output, 2, 40, 1));
//     ESP_ERROR_CHECK(dsps_add_f32_ae32((const float *)net2_3_output, net2_3_bias, net2_3_output, 2, 1, 1, 1));

//     output[0] = net2_3_output[0];
//     output[1] = net2_3_output[1];
// }

esp_err_t scritch_forward(float *accX, float *accY, float *accZ, bool* result) {// float *output) {
    esp_err_t err;
    err = dspm_mult_f32_ae32(net1_1_weight, (const float *)accX, net1_output, 20, 45, 1);
    if (err != ESP_OK) return err;
    err = dsps_add_f32_ae32((const float *)net1_output, net1_1_bias, net1_output, 20, 1, 1, 1);
    if (err != ESP_OK) return err;
    relu_inplace(net1_output, 20);

    err = dspm_mult_f32_ae32(net1_2_weight, (const float *)accY, (net1_output + 20), 20, 45, 1);
    if (err != ESP_OK) return err;
    err = dsps_add_f32_ae32((const float *)(net1_output + 20), net1_2_bias, (net1_output + 20), 20, 1, 1, 1);
    if (err != ESP_OK) return err;
    relu_inplace((net1_output + 20), 20);

    err = dspm_mult_f32_ae32(net1_3_weight, (const float *)accZ, (net1_output + 40), 20, 45, 1);
    if (err != ESP_OK) return err;
    err = dsps_add_f32_ae32((const float *)(net1_output + 40), net1_3_bias, (net1_output + 40), 20, 1, 1, 1);
    if (err != ESP_OK) return err;
    relu_inplace((net1_output + 40), 20);

    err = dspm_mult_f32_ae32(net2_1_weight, (const float *)net1_output, net2_1_output, 40, 60, 1);
    if (err != ESP_OK) return err;
    err = dsps_add_f32_ae32((const float *)net2_1_output, net2_1_bias, net2_1_output, 40, 1, 1, 1);
    if (err != ESP_OK) return err;
    relu_inplace(net2_1_output, 40);

    err = dspm_mult_f32_ae32(net2_3_weight, (const float *)net2_1_output, net2_3_output, 2, 40, 1);
    if (err != ESP_OK) return err;
    err = dsps_add_f32_ae32((const float *)net2_3_output, net2_3_bias, net2_3_output, 2, 1, 1, 1);
    if (err != ESP_OK) return err;

    *result = net2_3_output[1] > net2_3_output[0];
    return err;
}

void scritch_deinit() {
    free(net1_output);
    free(net2_1_output);
    free(net2_3_output);
}
