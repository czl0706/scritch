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
#include "esp_timer.h"
#include "driver/gpio.h"
#include "mpu6500.h"
#include "scritch_nn.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

static const char *TAG = "mpu6500-test";

#define BUTTON1 19
#define BUFFER_LEN 45 

#define COLLECT_INTERVAL 10
#define INFERENCE_COUNT 500 / COLLECT_INTERVAL

float accX[BUFFER_LEN], accY[BUFFER_LEN], accZ[BUFFER_LEN], output[2];

void detection_task(void *pvParameters) {
    int count = 0;
    acc_data_t accData;

    while (1) {
        // printf("%lld ", esp_timer_get_time() / 1000);
        ESP_ERROR_CHECK(mpu6500_get_acc(&accData));
        for (int i = 0; i < BUFFER_LEN - 1; i++) {
            accX[i] = accX[i + 1];
            accY[i] = accY[i + 1];
            accZ[i] = accZ[i + 1];
        }
        accX[BUFFER_LEN - 1] = accData.accX;
        accY[BUFFER_LEN - 1] = accData.accY;
        accZ[BUFFER_LEN - 1] = accData.accZ;

        count += 1;
        if (count >= INFERENCE_COUNT) {
            scritch_forward(accX, accY, accZ, output);
            // printf("%lld ", esp_timer_get_time() / 1000);
            // printf("%+f, %+f\n", output[0], output[1]);
            printf("%d\n", output[1] > output[0]);
            count = 0;
        }

        vTaskDelay(pdMS_TO_TICKS(COLLECT_INTERVAL));
    }
}

void app_main(void) {
    scritch_init();
    // measure the time of one forward pass
    // printf("%lld ", esp_timer_get_time() / 1000);
    // scritch_forward(accX, accY, accZ, output);
    // printf("%lld \n", esp_timer_get_time() / 1000);

    ESP_ERROR_CHECK(mpu6500_i2c_init());
    ESP_LOGI(TAG, "I2C initialized successfully");

    xTaskCreate(detection_task, "detection_task", 4096, NULL, 5, NULL);

    while (1) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

void data_collection_task(void *pvParameters) {
    acc_data_t accData;

    gpio_set_direction(BUTTON1, GPIO_MODE_INPUT);
    gpio_set_pull_mode(BUTTON1, GPIO_PULLUP_ONLY);

    printf("Start collection\n");
    
    while(true) {
        ESP_ERROR_CHECK(mpu6500_get_acc(&accData));

        printf("%+f, %+f, %+f, %d\n", accData.accX, accData.accY, accData.accZ, !gpio_get_level(BUTTON1));
        vTaskDelay(pdMS_TO_TICKS(3));
    }
}
