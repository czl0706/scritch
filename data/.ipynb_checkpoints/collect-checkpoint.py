import serial
import os

COM_PORT = 'COM6'       
BAUD_RATES = 115200     
ser = serial.Serial(COM_PORT, BAUD_RATES)   

# if not os.path.exists('./data'):
#     os.mkdir('./data')

# index = 0
f = open('data.csv', 'a')

try:
    while True:
        while ser.in_waiting:          
            data = ser.readline().decode()
            if 'i2cWriteReadNonStop' in data:
                print('Error')
                # f.close()
                # index += 1
                # f = open('./data/data' + str(index) + '.csv', 'a')
                continue
            f.write(data)
except KeyboardInterrupt:
    ser.close() 