import serial

COM_PORT = 'COM7'       
BAUD_RATES = 115200     
ser = serial.Serial(COM_PORT, BAUD_RATES)   