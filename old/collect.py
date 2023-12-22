import os
import keyboard
from sercomm import ser, ser_prepare

if not os.path.exists('./data'):
    os.mkdir('./data')

index = 0
f = open('./data/data.csv', 'a')
ser_prepare()

try:
    while True:
        if ser.in_waiting:          
            scratching = keyboard.is_pressed('space')
            
            data = ser.readline().decode().strip() + f', {scratching:d}'
            print(data)
            f.write(data + '\n')
            
except Exception as e:
    ser.close() 
    f.close()
    