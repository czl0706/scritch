#include <Arduino.h>
#include <BluetoothSerial.h> 
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#include <Ticker.h>

Adafruit_MPU6050 mpu;
// BluetoothSerial SerialBT; 
Ticker timer1;

void print_data() {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  // Serial.printf("%f, %f, %f, %f, %f, %f, %d\n", 
  //   a.acceleration.x, a.acceleration.y, a.acceleration.z, g.gyro.x, g.gyro.y, g.gyro.z, !digitalRead(32));

  Serial.printf("%f, %f, %f, %d\n", a.acceleration.x, a.acceleration.y, a.acceleration.z, !digitalRead(27));
}

void setup() {
  Serial.begin(115200);
  while (!mpu.begin());
  // Try to initialize!
  
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    while (1) {
      delay(10);
    }
  }

  // SerialBT.begin("ESP32_BT");
    
  mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
  mpu.setGyroRange(MPU6050_RANGE_250_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_260_HZ);

  mpu.setTemperatureStandby(true);
  mpu.setGyroStandby(true, true, true);
  mpu.setAccelerometerStandby(false, false, false);

  pinMode(27, INPUT_PULLUP);
  // Serial.print("Accelerometer range set to: ");
  // switch (mpu.getAccelerometerRange()) {
  // case MPU6050_RANGE_2_G:
  //   Serial.println("+-2G");
  //   break;
  // case MPU6050_RANGE_4_G:
  //   Serial.println("+-4G");
  //   break;
  // case MPU6050_RANGE_8_G:
  //   Serial.println("+-8G");
  //   break;
  // case MPU6050_RANGE_16_G:
  //   Serial.println("+-16G");
  //   break;
  // }

  // Serial.print("Gyro range set to: ");
  // switch (mpu.getGyroRange()) {
  // case MPU6050_RANGE_250_DEG:
  //   Serial.println("+- 250 deg/s");
  //   break;
  // case MPU6050_RANGE_500_DEG:
  //   Serial.println("+- 500 deg/s");
  //   break;
  // case MPU6050_RANGE_1000_DEG:
  //   Serial.println("+- 1000 deg/s");
  //   break;
  // case MPU6050_RANGE_2000_DEG:
  //   Serial.println("+- 2000 deg/s");
  //   break;
  // }

  // Serial.print("Filter bandwidth set to: ");
  // switch (mpu.getFilterBandwidth()) {
  // case MPU6050_BAND_260_HZ:
  //   Serial.println("260 Hz");
  //   break;
  // case MPU6050_BAND_184_HZ:
  //   Serial.println("184 Hz");
  //   break;
  // case MPU6050_BAND_94_HZ:
  //   Serial.println("94 Hz");
  //   break;
  // case MPU6050_BAND_44_HZ:
  //   Serial.println("44 Hz");
  //   break;
  // case MPU6050_BAND_21_HZ:
  //   Serial.println("21 Hz");
  //   break;
  // case MPU6050_BAND_10_HZ:
  //   Serial.println("10 Hz");
  //   break;
  // case MPU6050_BAND_5_HZ:
  //   Serial.println("5 Hz");
  //   break;
  // }

  delay(100);
  timer1.attach_ms(5, print_data);
}

// #define BUFFER_SIZE 100
// uint32_t count = 0;
// float buffer[BUFFER_SIZE];

void loop() {
  /* Get new sensor events with the readings */
  // sensors_event_t a, g, temp;
  // mpu.getEvent(&a, &g, &temp);

  // Serial.printf("%f, %f, %f, %f, %f, %f\n", 
  //   a.acceleration.x, a.acceleration.y, a.acceleration.z, g.gyro.x, g.gyro.y, g.gyro.z);

  // /* Print out the values */
  // Serial.print("Acceleration X: ");
  // Serial.print(a.acceleration.x);
  // Serial.print(", Y: ");
  // Serial.print(a.acceleration.y);
  // Serial.print(", Z: ");
  // Serial.print(a.acceleration.z);
  // Serial.println(" m/s^2");

  // Serial.print("Rotation X: ");
  // Serial.print(g.gyro.x);
  // Serial.print(", Y: ");
  // Serial.print(g.gyro.y);
  // Serial.print(", Z: ");
  // Serial.print(g.gyro.z);
  // Serial.println(" rad/s");

  // Serial.print("Temperature: ");
  // Serial.print(temp.temperature);
  // Serial.println(" degC");

  // Serial.println("");
  // Serial.print("x:");
  // Serial.print(a.acceleration.x);
  // Serial.print(",");
  // Serial.print("y:");
  // Serial.print(a.acceleration.y);
  // Serial.print(",");
  // Serial.print("z:");
  // Serial.println(a.acceleration.z);

  // Serial.println(digitalRead(32));

  // SerialBT.println(a.acceleration.z);
  // buffer[count++] = a.acceleration.z;
  // if (count == BUFFER_SIZE) {
  //   for (int i = 0; i < BUFFER_SIZE; i++) {
  //     SerialBT.print(buffer[i]);
  //     SerialBT.print(",");
  //   }
  //   SerialBT.println("");
  //   count = 0;
  // }

  // delay(500);
}


// // MPU-6050 Short Example Sketch
// // By Arduino User JohnChi
// // August 17, 2014
// // Public Domain
// #include <Arduino.h>
// #include <BluetoothSerial.h> 
// #include <Adafruit_MPU6050.h>
// #include <Adafruit_Sensor.h>
// #include <Wire.h>
// #include <Ticker.h>
// const int MPU_addr = 0x68; // I2C address of the MPU-6050
// int16_t AcX, AcY, AcZ;

// Ticker timer1;

// void print_data() {
//   Wire.beginTransmission(MPU_addr);
//   Wire.write(0x3B);  // starting with register 0x3B (ACCEL_XOUT_H)
//   Wire.endTransmission(false);
//   Wire.requestFrom(MPU_addr, 6); // request a total of 6 registers
//   int t = Wire.read();
//   AcX = (t << 8) | Wire.read(); // 0x3B (ACCEL_XOUT_H) & 0x3C (ACCEL_XOUT_L)
//   t = Wire.read();
//   AcY = (t << 8) | Wire.read(); // 0x3D (ACCEL_YOUT_H) & 0x3E (ACCEL_YOUT_L)
//   t = Wire.read();
//   AcZ = (t << 8) | Wire.read(); // 0x3F (ACCEL_ZOUT_H) & 0x40 (ACCEL_ZOUT_L)
//   Serial.print(AcX);
//   Serial.print(", "); Serial.print(AcY);
//   Serial.print(", "); Serial.println(AcZ);
// }

// void setup() {
//   Wire.begin();
//   Wire.beginTransmission(MPU_addr);
//   Wire.write(0x6B);  // PWR_MGMT_1 register
//   Wire.write(0);     // set to zero (wakes up the MPU-6050)
//   Wire.endTransmission(true);
//   Serial.begin(9600);
//   timer1.attach_ms(5, print_data);
// }

// void loop() {

// }