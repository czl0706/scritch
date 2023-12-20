import os
import numpy as np
from sercomm import ser, ser_prepare
from scipy.spatial.transform import Rotation as R
import math

import numpy as np

acceleration_data = np.array([0, 0, 0])
gravity_vector = np.array([0, 0, 1])

acc = np.zeros((3, 50), dtype=np.float32)

index = 0
ser_prepare()

idx = 0
try:
    while True:
        if ser.in_waiting:          
            data = list(map(float, ser.readline().decode().strip().split(',')[1:4]))
            acc[:, -1] = data
            # acc_x[-1], acc_y[-1], acc_z[-1] = data 
                        
            # acc_x = np.roll(acc_x, -1)
            # acc_y = np.roll(acc_y, -1)
            # acc_z = np.roll(acc_z, -1)
            acc = np.roll(acc, -1, axis=1) 
            idx += 1
            
            # print(data)
            if idx == 50:
                idx = 0
                
                acceleration_data = np.mean(acc, axis=1)
                
                # 歸一化加速度向量
                acceleration_vector = acceleration_data / np.linalg.norm(acceleration_data)

                # 計算旋轉軸
                rotation_axis = np.cross(acceleration_vector, gravity_vector)
                rotation_axis /= np.linalg.norm(rotation_axis)

                # 計算旋轉角度
                rotation_angle = np.arccos(np.dot(acceleration_vector, gravity_vector))

                # 構造旋轉矩陣
                rotation_matrix = R.from_rotvec(rotation_angle * rotation_axis).as_matrix()

                new_acc = rotation_matrix @ acc
                
                new_acc = new_acc - gravity_vector.T.reshape(3, 1)
                
                print(new_acc[:, -1], rotation_angle * 180 / math.pi)
                
except Exception as e:
    print(e)
    ser.close() 
    