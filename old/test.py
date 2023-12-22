import numpy as np
import torch
from utils import *
from sercomm import ser, ser_prepare

INTERVAL = 0.8

acc_x = np.zeros((1, WINDOW_LENGTH), dtype=np.float32)
acc_y = np.zeros((1, WINDOW_LENGTH), dtype=np.float32)
acc_z = np.zeros((1, WINDOW_LENGTH), dtype=np.float32)
index = 0

model = Scritch() #.cuda()
model.load_state_dict(torch.load('./models/model.pt'))
model.eval()

ser_prepare()
try:
    x = 0
    while True:
        if ser.in_waiting:          
            data = ser.readline().decode()
            x += 1
            if x == DS_FACTOR:
                x = 0   
                try:
                    new_val = list(map(float, data.split(',')[:-1]))

                    acc_x = np.roll(acc_x, -1)
                    acc_y = np.roll(acc_y, -1)
                    acc_z = np.roll(acc_z, -1)
                    acc_x[0][-1], acc_y[0][-1], acc_z[0][-1] = new_val
                    # print(new_val)
                except:
                    continue
                index += 1
                if index == int(INTERVAL/MODL_SAMPLING_PERIOD):
                    with torch.no_grad():
                        scratching = model(torch.from_numpy(proc_data(acc_x, 
                                                                      acc_y, 
                                                                      acc_z))
                                        ).argmax(dim=1).item()
                        print("Are you scratching?" if scratching else "Everything looks fine")
                    
                    index = 0
except KeyboardInterrupt:
    ser.close() 