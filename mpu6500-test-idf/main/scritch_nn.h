#ifndef __SCRITCH_NN_H__
#define __SCRITCH_NN_H__

#include "model.h"
#include "dspm_mult.h"
#include "dsps_add.h"

// typedef struct {
//     float *weights;
//     float *bias;
//     int input_size;
//     int output_size;
// } LinearLayer;

void scritch_init();
void scritch_forward(float *accX, float *accY, float *accZ, float *output);

#endif