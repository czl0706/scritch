import numpy as np
import torch
from model import *
from serial_port import ser, ser_prepare

INTERVAL = 0.5
THRESHOLD = 0.5

x = np.zeros((1, int(WINDOW_LENGTH/SAMPLING_PERIOD)), dtype=np.float32)
# x = [0] * 200
index = 0

model = Scritch() #.cuda()
model.load_state_dict(torch.load('./models/model.pt'))
model.eval()

ser_prepare()
try:
    while True:
        if ser.in_waiting:          
            data = ser.readline().decode()
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
                    if model(torch.from_numpy(x)).item() > THRESHOLD:
                        print('Are you scratching?')
                    else:
                        print('Everything looks fine')
                index = 0
except KeyboardInterrupt:
    ser.close() 