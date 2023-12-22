import os
import keyboard
from sercomm import ser, ser_prepare
import numpy as np
import time

path_name = './data'
data_name = 'data_{}.txt' 
label_name = 'label_{}.txt' 

if not os.path.exists(path_name):
    os.mkdir(path_name)

# Check the smallest serial number
for i in range(1, 100):
    if not os.path.exists(os.path.join(path_name, data_name.format(i))):
        serial_number = i
        break

data_name = data_name.format(serial_number)
label_name = label_name.format(serial_number) 

f = open(os.path.join(path_name, data_name), 'a')
f_lbl = open(os.path.join(path_name, label_name), 'a')

buffer = ''
scratching_buffer = np.zeros(50, dtype=bool)

last_time = time.time()
ser_prepare()

try:
    while True:
        if time.time() - last_time > 0.01:
            last_time = time.time()
            scratching = keyboard.is_pressed('F10')
            scratching_buffer[:-1] = scratching_buffer[1:]
            scratching_buffer[-1] = scratching
            
        if ser.in_waiting:          
            data = ser.readline().decode().strip() # + f', {scratching:d}'
            # print(data)
            f.write(data + '\n')
            if data == '':
                prob = np.mean(scratching_buffer)
                print(prob)
                f_lbl.write(f'{prob}\n')
        elif keyboard.is_pressed('F9'):
            break
        
except Exception as e:
    ...
    
ser.close() 
f.close()