#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "sdkconfig.h"
#include "mpu6050.h"
#include "esp_timer.h"
#define _USE_MATH_DEFINES
#include <math.h>

#include "scritch_nn.h"

#define I2C_MASTER_SCL_IO 5       /*!< gpio number for I2C master clock */
#define I2C_MASTER_SDA_IO 4       /*!< gpio number for I2C master data  */
#define I2C_MASTER_NUM I2C_NUM_0  /*!< I2C port number for master dev */
#define I2C_MASTER_FREQ_HZ 400000 /*!< I2C master clock frequency */

static const char *TAG = "mpu6050-test";
static mpu6050_handle_t mpu6050 = NULL;

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
static void i2c_sensor_mpu6050_init(void)
{
    esp_err_t ret;

    i2c_bus_init();
    mpu6050 = mpu6050_create(I2C_MASTER_NUM, MPU6050_I2C_ADDRESS);

    ret = mpu6050_config(mpu6050, ACCE_FS_2G, GYRO_FS_1000DPS);

    ret = mpu6050_wake_up(mpu6050);
}

static inline float invSqrt(float x) 
{
    float half_x = 0.5f * x;
    float y = x;
    long i = *(long *)&y;
    i = 0x5f3759df - (i >> 1);
    y = *(float *)&i;
    y = y * (1.5f - (half_x * y * y));
    return y;
}

static inline float invNorm(float x, float y, float z)
{
    return invSqrt(x * x + y * y + z * z);
}

static inline void transform(float *accX, float *accY, float *accZ, float sin, float cos, float rotx, float roty)
{
    float x = *accX;
    float y = *accY;
    float z = *accZ;
    *accX = x * cos + roty * z * sin;
    *accY = y * cos - rotx * z * sin;
    *accZ = z * cos + (rotx * y - roty * x) * sin;
}

#define WINDOW_SIZE 50
#define STRIDE_SIZE 25

#define OUTPUT_WINDOW_SIZE 5
#define OUTPUT_THRESHOLD 3

float acc[WINDOW_SIZE][3];
float acc_trans[WINDOW_SIZE * 3];

// float gyro_diff[WINDOW_SIZE][3];
// float gyro_diff_buf[WINDOW_SIZE][3];

SemaphoreHandle_t xAccDataSemaphore;

static void collect_task()
{
    mpu6050_acce_value_t acce;
    // mpu6050_gyro_value_t gyro;
    float accX, accY, accZ;
    // float gyroX, gyroY, gyroZ;
    // float gyroX_prev = 0, gyroY_prev = 0, gyroZ_prev = 0;
    int stride_count = 0, idx = 0;

    const TickType_t xFrequency = pdMS_TO_TICKS(10);
    TickType_t xLastWakeTime = xTaskGetTickCount();

    for (int i = 0; i < WINDOW_SIZE; i++) {
        for (int j = 0; j < 3; j++) {
            acc_trans[i + j * 50] = 0;
            acc[i][j] = 0;
        }
    }

    printf("Start\n");
    while (1)
    {
        ESP_ERROR_CHECK(mpu6050_get_acce(mpu6050, &acce));
        // ESP_ERROR_CHECK(mpu6050_get_gyro(mpu6050, &gyro));

        accX = acce.acce_y;
        accY = acce.acce_x;
        accZ = -acce.acce_z;

        // gyroX = gyro.gyro_y;
        // gyroY = gyro.gyro_x;
        // gyroZ = -gyro.gyro_z;

        acc[idx][0] = accX;
        acc[idx][1] = accY;
        acc[idx][2] = accZ;

        // gyro_diff[idx][0] = gyroX - gyroX_prev;
        // gyro_diff[idx][1] = gyroY - gyroY_prev;
        // gyro_diff[idx][2] = gyroZ - gyroZ_prev;

        // gyroX_prev = gyroX;
        // gyroY_prev = gyroY;
        // gyroZ_prev = gyroZ;

        idx++;
        stride_count++;

        // a stride is completed, transport data to transform task
        // copy data from acc to accrot
        if (stride_count >= STRIDE_SIZE) {
            for (int i = 0; i < WINDOW_SIZE; i++) {
                for (int j = 0; j < 3; j++) {
                    acc_trans[i + j * 50] = acc[i][j];
                    // gyro_diff_buf[i][j] = gyro_diff[i][j];
                }
            }
            xSemaphoreGive(xAccDataSemaphore);

            if (idx >= WINDOW_SIZE) {
                // shift acc
                for (int i = 0; i < WINDOW_SIZE - STRIDE_SIZE; i++) {
                    acc[i][0] = acc[i + STRIDE_SIZE][0];
                    acc[i][1] = acc[i + STRIDE_SIZE][1];
                    acc[i][2] = acc[i + STRIDE_SIZE][2];
                }
                idx = WINDOW_SIZE - STRIDE_SIZE;
            }
            stride_count = 0;
        }

        // printf("%8lld, %+8f, %+8f, %+8f, %+11f, %+11f, %+11f\n", esp_timer_get_time() / 1000,
        //     accX, accY, accZ, gyroX, gyroY, gyroZ);

        // printf("%8lld, %+8f, %+8f, %+8f\n", esp_timer_get_time() / 1000, accX, accY, accZ);

        vTaskDelayUntil(&xLastWakeTime, xFrequency);
    }
}

static void transform_forward_task()
{
    float acc_invNorm, rot_invNorm;
    float accX_norm = 0, accY_norm = 0, accZ_norm = 0;

    float cos_rot_angle, sin_rot_angle;
    float rot_x, rot_y;

    scritch_init(acc_trans);

    uint16_t record = 0;

    while (1)
    {
        if (xSemaphoreTake(xAccDataSemaphore, portMAX_DELAY)) {
            accX_norm = 0;
            accY_norm = 0;
            accZ_norm = 0;

            for (int i = 0; i < WINDOW_SIZE; i++) {
                accX_norm += acc_trans[i + 50 * 0];
                accY_norm += acc_trans[i + 50 * 1];
                accZ_norm += acc_trans[i + 50 * 2];
            }

            accX_norm /= WINDOW_SIZE;
            accY_norm /= WINDOW_SIZE;
            accZ_norm /= WINDOW_SIZE;

            for (int i = 0; i < WINDOW_SIZE; i++) {
                acc_trans[i + 50 * 0] -= accX_norm;
                acc_trans[i + 50 * 1] -= accY_norm;
                acc_trans[i + 50 * 2] -= accZ_norm;
            }

            acc_invNorm = invNorm(accX_norm, accY_norm, accZ_norm);
            
            accX_norm *= acc_invNorm;
            accY_norm *= acc_invNorm;
            accZ_norm *= acc_invNorm;

            rot_x = accY_norm;
            rot_y = -accX_norm;

            rot_invNorm = invNorm(rot_x, rot_y, 0);
            rot_x *= rot_invNorm;
            rot_y *= rot_invNorm;

            cos_rot_angle = accZ_norm;
            sin_rot_angle = sqrt(1 - accZ_norm * accZ_norm);

            for (int i = 0; i < WINDOW_SIZE; i++) {
                transform(&acc_trans[i + 50 * 0], &acc_trans[i + 50 * 1], &acc_trans[i + 50 * 2], 
                                    sin_rot_angle, cos_rot_angle, rot_x, rot_y);
                // printf("%+.4f, %+.4f, %+.4f\n", acc_trans[i][0], acc_trans[i][1], acc_trans[i][2]);
            }
            
            record = record << 1 | scritch_forward();
            uint16_t count = 0, tmp = record;

            // printf("%d\n", record);  

            for (int i = 0; i < OUTPUT_WINDOW_SIZE; i++) {
                count += tmp & 1;
                tmp >>= 1;
            }
            
            if (count >= OUTPUT_THRESHOLD) {
                printf("1\n");
            } else {
                printf("0\n");
            }
            // printf("%d\n", scritch_forward());


            // accX_norm = 0;
            // accY_norm = 0;
            // accZ_norm = 0;

            // for (int i = 0; i < WINDOW_SIZE; i++) {
            //     accX_norm += acc_trans[i][0];
            //     accY_norm += acc_trans[i][1];
            //     accZ_norm += acc_trans[i][2];
            // }

            // accX_norm /= WINDOW_SIZE;
            // accY_norm /= WINDOW_SIZE;
            // accZ_norm /= WINDOW_SIZE;

            // acc_invNorm = invNorm(accX_norm, accY_norm, accZ_norm);
            
            // accX_norm *= acc_invNorm;
            // accY_norm *= acc_invNorm;
            // accZ_norm *= acc_invNorm;

            // for (int i = 0; i < WINDOW_SIZE; i++) {                  
            //     printf("%+.4f, %+.4f, %+.4f\n", acc_trans[i][0] - accX_norm, 
            //                                     acc_trans[i][1] - accY_norm, 
            //                                     acc_trans[i][2] - accZ_norm);

            //     // printf("%+.4f, %+.4f, %+.4f\n", gyro_diff_buf[i][0] / 20, 
            //     //                                 gyro_diff_buf[i][1] / 20, 
            //     //                                 gyro_diff_buf[i][2] / 20);
            // }
            // // printf("%lld\n", esp_timer_get_time() / 1000);
            // printf("\n");
        }
    }
}

void app_main()
{
    esp_err_t ret;
    uint8_t mpu6050_deviceid;

    i2c_sensor_mpu6050_init();

    ret = mpu6050_get_deviceid(mpu6050, &mpu6050_deviceid);

    xAccDataSemaphore = xSemaphoreCreateBinary();
    if (xAccDataSemaphore == NULL) {
        printf("Failed to create semaphore\n");
    }

    xTaskCreate(collect_task, "collect_task", 8192, NULL, 5, NULL);
    vTaskDelay(pdMS_TO_TICKS(500));
    xTaskCreate(transform_forward_task, "transform_forward_task", 8192, NULL, 5, NULL);

    while (1)
    {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }

    mpu6050_delete(mpu6050);
    ret = i2c_driver_delete(I2C_MASTER_NUM);
}
