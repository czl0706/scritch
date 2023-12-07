#include "mpu6500.h"

/**
 * @brief Read a sequence of bytes from a MPU6500 sensor registers
 */
static inline esp_err_t mpu6500_register_read(uint8_t reg_addr, uint8_t *data, size_t len)
{
    return i2c_master_write_read_device(I2C_MASTER_NUM, MPU6500_SENSOR_ADDR, &reg_addr, 1, data, len, I2C_MASTER_TIMEOUT_MS / portTICK_PERIOD_MS);
}

// /**
//  * @brief Write a byte to a MPU6500 sensor register
//  */
// static esp_err_t mpu6500_register_write_byte(uint8_t reg_addr, uint8_t data)
// {
//     int ret;
//     uint8_t write_buf[2] = {reg_addr, data};

//     ret = i2c_master_write_to_device(I2C_MASTER_NUM, MPU6500_SENSOR_ADDR, write_buf, sizeof(write_buf), I2C_MASTER_TIMEOUT_MS / portTICK_PERIOD_MS);

//     return ret;
// }

/**
 * @brief i2c master initialization
 */
esp_err_t mpu6500_i2c_init(void)
{
    int i2c_master_port = I2C_MASTER_NUM;

    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = I2C_MASTER_SDA_IO,
        .scl_io_num = I2C_MASTER_SCL_IO,
        .sda_pullup_en = I2C_INTERNAL_PULLUP,
        .scl_pullup_en = I2C_INTERNAL_PULLUP,
        .master.clk_speed = I2C_MASTER_FREQ_HZ,
    };

    i2c_param_config(i2c_master_port, &conf);

    return i2c_driver_install(i2c_master_port, conf.mode, I2C_MASTER_RX_BUF_DISABLE, I2C_MASTER_TX_BUF_DISABLE, 0);
}

esp_err_t mpu6500_get_accZ(float *accZ)
{
    uint8_t data[2];
    int16_t rawZ;
    int ret;

    ret = mpu6500_register_read(MPU6500_ACCEL_ZOUT_H, data, 2);
    rawZ = (data[0] << 8 | data[1]);
    *accZ = rawZ / accel_scale - accZ_offset;
    return ret;
}

esp_err_t mpu6500_get_acc(acc_data_t *acc)
{
    uint8_t data[6];
    int16_t rawX, rawY, rawZ;
    int ret;

    ret = mpu6500_register_read(MPU6500_ACCEL_XOUT_H, data, 6);
    rawX = (data[0] << 8 | data[1]);
    rawY = (data[2] << 8 | data[3]);
    rawZ = (data[4] << 8 | data[5]);
    acc->accX = rawX / accel_scale - accX_offset;
    acc->accY = rawY / accel_scale - accY_offset;
    acc->accZ = rawZ / accel_scale - accZ_offset;
    return ret;
}