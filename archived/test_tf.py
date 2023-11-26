import numpy as np
import tensorflow as tf
import keras
from serial_port import ser

model = keras.models.load_model('../models/model.h5')

x = np.zeros((1, 200), dtype=np.float32)
index = 0

try:
    while True:
        while ser.in_waiting:          
            data = ser.readline().decode()
            if 'i2cWriteReadNonStop' in data:
                continue
            # x[index] = float(data.split(',')[2])
            # index += 1
            # if index == 200:
            #     index = 0
            #     print(model.predict(tf.convert_to_tensor([x])))
            try:
                new_val = float(data.split(',')[2])
                x[0][-1] = new_val
                x = np.roll(x, -1)
            except:
                continue
            index += 1
            if index == 50:
                if model.predict(tf.convert_to_tensor(x), verbose=0).squeeze() >= 0.6:
                    print('Are you scratching?')
                # print(model.predict(tf.convert_to_tensor(x), verbose=0).squeeze())
                index = 0
except KeyboardInterrupt:
    ser.close() 