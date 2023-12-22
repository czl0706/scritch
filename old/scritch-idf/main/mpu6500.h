#ifndef __MPU6500_H__
#define __MPU6500_H__

#include "driver/i2c.h"

#define I2C_MASTER_SCL_IO           CONFIG_I2C_MASTER_SCL      /*!< GPIO number used for I2C master clock */
#define I2C_MASTER_SDA_IO           CONFIG_I2C_MASTER_SDA      /*!< GPIO number used for I2C master data  */
#define I2C_MASTER_NUM              0                          /*!< I2C master i2c port number, the number of i2c peripheral interfaces available will depend on the chip */
#define I2C_MASTER_FREQ_HZ          400000                     /*!< I2C master clock frequency */
#define I2C_MASTER_TX_BUF_DISABLE   0                          /*!< I2C master doesn't need buffer */
#define I2C_MASTER_RX_BUF_DISABLE   0                          /*!< I2C master doesn't need buffer */
#define I2C_MASTER_TIMEOUT_MS       1000
#define I2C_INTERNAL_PULLUP         false                       // Set to false if external pull-up resistor is used 

#define MPU6500_SENSOR_ADDR                 0x68        /*!< Slave address of the MPU6500 sensor */
#define MPU6500_WHO_AM_I_REG_ADDR           0x70        /*!< Register addresses of the "who am I" register */

#define MPU6500_ACCEL_XOUT_H                0x3B
#define MPU6500_ACCEL_ZOUT_H                0x3F

#define MPU6500_PWR_MGMT_1_REG_ADDR         0x6B        /*!< Register addresses of the power managment register */
#define MPU6500_RESET_BIT                   7

static const float accel_scale = 16384;
static const float accX_offset = 0.03;
static const float accY_offset = 0;
static const float accZ_offset = 0.13;

typedef struct acc_data {
    float accX;
    float accY;
    float accZ;
} acc_data_t;

esp_err_t mpu6500_i2c_init(void);
esp_err_t mpu6500_get_accZ(float *accZ);
esp_err_t mpu6500_get_acc(acc_data_t *acc);

#endif