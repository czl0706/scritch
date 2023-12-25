#ifndef __I2C_MPU6050_H__
#define __I2C_MPU6050_H__

#include "mpu6050.h"

#define I2C_MASTER_SCL_IO 5       /*!< gpio number for I2C master clock */
#define I2C_MASTER_SDA_IO 4       /*!< gpio number for I2C master data  */
#define I2C_MASTER_NUM I2C_NUM_0  /*!< I2C port number for master dev */
#define I2C_MASTER_FREQ_HZ 400000 /*!< I2C master clock frequency */

mpu6050_handle_t i2c_sensor_mpu6050_init(void);

#endif