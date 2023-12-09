#include <stdio.h>
#include "esp_log.h"
#include "esp_timer.h"
#include "driver/gpio.h"
#include "mpu6500.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

static const char *TAG = "scritch-collect";

#define LABEL_BUTTON 19

#define LED_BLINK 0
#define LED_PIN 2

static inline void led_change_state(bool state) {
    gpio_set_level(LED_PIN, state);
}

static inline void led_init(void) {
    gpio_reset_pin(LED_PIN);
    gpio_set_direction(LED_PIN, GPIO_MODE_OUTPUT);
}

static void collection_task(void *pvParameters) {
    acc_data_t accData;
    bool result;

    gpio_set_direction(LABEL_BUTTON, GPIO_MODE_INPUT);
    gpio_set_pull_mode(LABEL_BUTTON, GPIO_PULLUP_ONLY);

    printf("Start collection\n");
    
    while(true) {
        ESP_ERROR_CHECK(mpu6500_get_acc(&accData));
        result = !gpio_get_level(LABEL_BUTTON);

        printf("%+f, %+f, %+f, %d\n", accData.accX, accData.accY, accData.accZ, result);
#if LED_BLINK
            led_change_state(result);
#endif
        vTaskDelay(pdMS_TO_TICKS(3));
    }
}

void app_main(void) {
    led_init();

    ESP_ERROR_CHECK(mpu6500_i2c_init());
    ESP_LOGI(TAG, "I2C initialized successfully");

    xTaskCreate(collection_task, "collection_task", 4096, NULL, 5, NULL);

    while (1) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}