import numpy as np
import torch
import serial
from model import *
from serial_port import ser

INTERVAL = 0.5

x = np.zeros((1, int(WINDOW_LENGTH/SAMPLING_PERIOD)), dtype=np.float32)
# x = [0] * 200
index = 0

model = Scritch() #.cuda()
model.load_state_dict(torch.load('./models/model.pt'))
model.eval()

try:
    while True:
        while ser.in_waiting:          
            data = ser.readline().decode()
            if 'i2cWriteReadNonStop' in data:
                # print('Error')
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
            if index == int(INTERVAL/SAMPLING_PERIOD):
                with torch.no_grad():
                    # print(model(torch.from_numpy(x)).item())
                    if model(torch.from_numpy(x)).item() > 0.5:
                        print('Are you scratching?')
                    else:
                        print('Everything looks fine')
                index = 0
except KeyboardInterrupt:
    ser.close() 