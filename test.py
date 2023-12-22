import numpy as np
import torch
from utils import *
from sercomm import ser, ser_prepare
from io import StringIO

model = Scritch() #.cuda()
model.load_state_dict(torch.load('./models/model.pt'))
model.eval()

ser_prepare()
try:
    data = ''
    while True:
        if ser.in_waiting:          
            in_data = ser.readline().decode().strip() # + f', {scratching:d}'
            data += in_data + '\n'
            if in_data == '' and data != '':
                inp_feat = np.genfromtxt(StringIO(data), delimiter=",").T.astype('float32')
                with torch.no_grad():
                    scratching = model(torch.from_numpy(inp_feat)).argmax(dim=1).item()
                    print("Are you scratching?" if scratching else "Everything looks fine")
                data = ''
                
except KeyboardInterrupt:
    ser.close() 