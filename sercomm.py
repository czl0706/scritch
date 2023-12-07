import serial
from time import sleep

COM_PORT = 'COM6'       
BAUD_RATES = 115200     
ser = serial.Serial(COM_PORT, BAUD_RATES)   

def ser_prepare():
    ser.setDTR(False)
    sleep(1)
    ser.flushInput()
    ser.setDTR(True)
    while True:
        if ser.in_waiting:          
            data = ser.readline().decode()
            if 'Start collection\r\n' == data: 
                return