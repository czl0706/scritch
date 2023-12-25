#ifndef __BLE_MAIN_H__
#define __BLE_MAIN_H__

#include "esp_log.h"
#include "nvs_flash.h"
#include "freertos/FreeRTOSConfig.h"
/* BLE */
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "host/ble_hs.h"
#include "host/util/util.h"
#include "console/console.h"
#include "services/gap/ble_svc_gap.h"
#include "blehr_sens.h"

extern bool notify_state;
void ble_main(void);
void update_my_characteristic_value(uint8_t new_value);

#endif