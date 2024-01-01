#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "sdkconfig.h"
#include "esp_timer.h"
#define _USE_MATH_DEFINES
#include <math.h>

#include "i2c_mpu6050.h"
#include "scritch_nn.h"
#include "math_utils.h"
#include "ble_main.h"

#define WINDOW_SIZE 50
#define STRIDE_SIZE 25

#define OUTPUT_WINDOW_SIZE 5
#define OUTPUT_THRESHOLD 3

#define USE_LED1_INDICATOR 1
#define USE_LED2_INDICATOR 0
#define LED1_PIN 12
#define LED2_PIN 13

static const char *tag = "scritch-main";
static mpu6050_handle_t mpu6050 = NULL;

float acc[WINDOW_SIZE][3];
float acc_trans[WINDOW_SIZE * 3];

// float gyro_diff[WINDOW_SIZE][3];
// float gyro_diff_buf[WINDOW_SIZE][3];

SemaphoreHandle_t xAccDataSemaphore;

static void led1_change_state(bool state) {
    gpio_set_level(LED1_PIN, state);
}

static void led2_change_state(bool state) {
    gpio_set_level(LED2_PIN, state);
}

static void led_init(void) {
    gpio_reset_pin(LED1_PIN);
    gpio_set_direction(LED1_PIN, GPIO_MODE_OUTPUT);
    gpio_set_level(LED1_PIN, 0);

    gpio_reset_pin(LED2_PIN);
    gpio_set_direction(LED2_PIN, GPIO_MODE_OUTPUT);
    gpio_set_level(LED2_PIN, 0);
}

static void get_acc_task()
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

    ESP_LOGI(tag, "Starting...\n");
    // printf("Start\n");
    while (1)
    {
        ESP_ERROR_CHECK(mpu6050_get_acce(mpu6050, &acce));

        accX = acce.acce_y;
        accY = acce.acce_x;
        accZ = -acce.acce_z;

        acc[idx][0] = accX;
        acc[idx][1] = accY;
        acc[idx][2] = accZ;

        idx++;
        stride_count++;

        // a stride is completed, transport data to transform task
        // copy data from acc to accrot
        if (stride_count >= STRIDE_SIZE) {
            for (int i = 0; i < WINDOW_SIZE; i++) {
                for (int j = 0; j < 3; j++) {
                    acc_trans[i + j * 50] = acc[i][j];
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

static void trans_pred_task()
{
    float acc_invNorm, rot_invNorm;
    float accX_norm = 0, accY_norm = 0, accZ_norm = 0;

    float cos_rot_angle, sin_rot_angle;
    float rot_x, rot_y;

    bool scratching;

    uint16_t record = 0;

    scritch_init(acc_trans);

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

            scratching = (count >= OUTPUT_THRESHOLD);
            
            ESP_LOGI(tag, "Status: %d\n", scratching);
            // printf("%d\n", scratching);

            #if USE_LED1_INDICATOR
                led1_change_state(scratching);
            #endif

            #if USE_LED2_INDICATOR
                led2_change_state(scratching);
            #endif

            if (notify_state) {
                update_my_characteristic_value(scratching);
            }
            

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
    // uint8_t mpu6050_deviceid;

    mpu6050 = i2c_sensor_mpu6050_init();

    // mpu6050_get_deviceid(mpu6050, &mpu6050_deviceid);

    led_init();

    ble_main();

    xAccDataSemaphore = xSemaphoreCreateBinary();
    if (xAccDataSemaphore == NULL) {
        ESP_LOGW(tag, "Failed to create semaphore\n");
        // printf("Failed to create semaphore\n");
    }

    xTaskCreate(get_acc_task, "get_acc_task", 8192, NULL, 5, NULL);
    vTaskDelay(pdMS_TO_TICKS(500));
    xTaskCreate(trans_pred_task, "trans_pred_task", 8192, NULL, 5, NULL);

    while (1) 
    {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }

    mpu6050_delete(mpu6050);
    i2c_driver_delete(I2C_MASTER_NUM);
}
