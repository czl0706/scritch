import os

from serial_port import ser

if not os.path.exists('./data'):
    os.mkdir('./data')

index = 0
f = open('./data/data.csv', 'a')

try:
    while True:
        while ser.in_waiting:          
            data = ser.readline().decode()
            if 'i2cWriteReadNonStop' in data:
                print(f'Error {index}')
                index += 1
                continue
            f.write(data)
except KeyboardInterrupt:
    ser.close() 