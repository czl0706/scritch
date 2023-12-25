#ifndef __SCRITCH_NN_H__
#define __SCRITCH_NN_H__

#include "dspm_mult.h"
#include "dsps_add.h"
#include <stdbool.h>

void scritch_init(float *_input_data);
void scritch_deinit();
bool scritch_forward();

#endif