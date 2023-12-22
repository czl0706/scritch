#ifndef __SCRITCH_NN_H__
#define __SCRITCH_NN_H__

#include "model_weights.h"
#include "dspm_mult.h"
#include "dsps_add.h"
#include <stdbool.h>

void scritch_init();
void scritch_deinit();
// void scritch_forward(float *accX, float *accY, float *accZ, float *output);
// esp_err_t scritch_forward(float *accX, float *accY, float *accZ, float *output);
esp_err_t scritch_forward(float *accX, float *accY, float *accZ, bool *result);

#endif