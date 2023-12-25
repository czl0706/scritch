#include "i2c_mpu6050.h"

/**
 * @brief i2c master initialization
 */
static void i2c_bus_init(void)
{
    i2c_config_t conf;
    conf.mode = I2C_MODE_MASTER;
    conf.sda_io_num = (gpio_num_t)I2C_MASTER_SDA_IO;
    conf.sda_pullup_en = GPIO_PULLUP_DISABLE;
    conf.scl_io_num = (gpio_num_t)I2C_MASTER_SCL_IO;
    conf.scl_pullup_en = GPIO_PULLUP_DISABLE;
    conf.master.clk_speed = I2C_MASTER_FREQ_HZ;
    conf.clk_flags = I2C_SCLK_SRC_FLAG_FOR_NOMAL;

    esp_err_t ret = i2c_param_config(I2C_MASTER_NUM, &conf);

    ret = i2c_driver_install(I2C_MASTER_NUM, conf.mode, 0, 0, 0);
}

/**
 * @brief i2c master initialization
 */
mpu6050_handle_t i2c_sensor_mpu6050_init(void)
{
    // esp_err_t ret;
    mpu6050_handle_t mpu6050 = NULL;

    i2c_bus_init();
    mpu6050 = mpu6050_create(I2C_MASTER_NUM, MPU6050_I2C_ADDRESS);

    // ret = 
    mpu6050_config(mpu6050, ACCE_FS_2G, GYRO_FS_1000DPS);

    // ret = 
    mpu6050_wake_up(mpu6050);

    return mpu6050;
}