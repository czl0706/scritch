/* i2c - Simple example

   Simple I2C example that shows how to initialize I2C
   as well as reading and writing from and to registers for a sensor connected over I2C.

   The sensor used in this example is a MPU6500 inertial measurement unit.

   For other examples please check:
   https://github.com/espressif/esp-idf/tree/master/examples

   See README.md file to get detailed usage of this example.

   This example code is in the Public Domain (or CC0 licensed, at your option.)

   Unless required by applicable law or agreed to in writing, this
   software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
   CONDITIONS OF ANY KIND, either express or implied.
*/
#include <stdio.h>
#include "esp_log.h"
#include "driver/i2c.h"
#include "esp_timer.h"

static const char *TAG = "mpu6500-test";

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

#define BUTTON1                             19

// #define LOG_LOCAL_LEVEL                     ESP_LOG_INFO

/**
 * @brief Read a sequence of bytes from a MPU6500 sensor registers
 */
static esp_err_t mpu6500_register_read(uint8_t reg_addr, uint8_t *data, size_t len)
{
    return i2c_master_write_read_device(I2C_MASTER_NUM, MPU6500_SENSOR_ADDR, &reg_addr, 1, data, len, I2C_MASTER_TIMEOUT_MS / portTICK_PERIOD_MS);
}

/**
 * @brief Write a byte to a MPU6500 sensor register
 */
static esp_err_t mpu6500_register_write_byte(uint8_t reg_addr, uint8_t data)
{
    int ret;
    uint8_t write_buf[2] = {reg_addr, data};

    ret = i2c_master_write_to_device(I2C_MASTER_NUM, MPU6500_SENSOR_ADDR, write_buf, sizeof(write_buf), I2C_MASTER_TIMEOUT_MS / portTICK_PERIOD_MS);

    return ret;
}

/**
 * @brief i2c master initialization
 */
static esp_err_t i2c_master_init(void)
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

const float accel_scale = 16384;
const float accX_offset = 0.03;
const float accY_offset = 0;
const float accZ_offset = 0.13;

static esp_err_t mpu6500_get_accZ(float *accZ)
{
    uint8_t data[2];
    int16_t rawZ;
    int ret;

    ret = mpu6500_register_read(MPU6500_ACCEL_ZOUT_H, data, 2);
    rawZ = (data[0] << 8 | data[1]);
    *accZ = rawZ / accel_scale - accZ_offset;
    return ret;
}

typedef struct acc_data {
    float accX;
    float accY;
    float accZ;
} acc_data_t;

static esp_err_t mpu6500_get_acc(acc_data_t *acc)
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

void app_main(void)
{
    uint8_t data[2];
    acc_data_t accData;

    gpio_set_direction(BUTTON1, GPIO_MODE_INPUT);
    gpio_set_pull_mode(BUTTON1, GPIO_PULLUP_ONLY);

    ESP_ERROR_CHECK(i2c_master_init());
    ESP_LOGI(TAG, "I2C initialized successfully");

    /* Read the MPU6500 WHO_AM_I register, on power up the register should have the value 0x71 */
    ESP_ERROR_CHECK(mpu6500_register_read(MPU6500_WHO_AM_I_REG_ADDR, data, 1));
    // ESP_LOGI(TAG, "WHO_AM_I = %X", data[0]);

    /* Demonstrate writing by reseting the MPU6500 */
    // ESP_ERROR_CHECK(mpu6500_register_write_byte(MPU6500_PWR_MGMT_1_REG_ADDR, 1 << MPU6500_RESET_BIT));

    while(true) {
        // ESP_ERROR_CHECK(mpu6500_register_read(MPU6500_ACCEL_ZOUT_H, data, 2));
        // rawZ = (data[0] << 8 | data[1]);
        // accZ = rawZ / accel_scale - accZ_offset;
        // ESP_ERROR_CHECK(mpu6500_get_accZ(&accZ));
        ESP_ERROR_CHECK(mpu6500_get_acc(&accData));

        // ESP_LOGI(TAG, "AccZ: %f", accZ);
        // printf("%lld %f %f %f\n", esp_timer_get_time() / 1000, accData.accX, accData.accY, accData.accZ);
        printf("%f %f %f %d\n", accData.accX, accData.accY, accData.accZ, gpio_get_level(BUTTON1));
        vTaskDelay(pdMS_TO_TICKS(3));
    }

    ESP_ERROR_CHECK(i2c_driver_delete(I2C_MASTER_NUM));
}
