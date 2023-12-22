from scipy.spatial.transform import Rotation as R
import math

import numpy as np

acceleration_data = np.array([0, 0, 0])
gravity_vector = np.array([0, 0, 1])

# acc = np.zeros((3, 50), dtype=np.float32)
np.random.seed(0)
acc = np.random.randn(3, 50).astype(np.float32)

acceleration_data = np.mean(acc, axis=1)

print(acceleration_data)
                
# 歸一化加速度向量
acceleration_vector = acceleration_data / np.linalg.norm(acceleration_data)

print(acceleration_vector)

# 計算旋轉軸
rotation_axis = np.cross(acceleration_vector, gravity_vector)
rotation_axis /= np.linalg.norm(rotation_axis)

print(rotation_axis)

# 計算旋轉角度
rotation_angle = np.arccos(np.dot(acceleration_vector, gravity_vector))

print(rotation_angle)

# 構造旋轉矩陣
rotation_matrix = R.from_rotvec(rotation_angle * rotation_axis).as_matrix()

# print(rotation_angle * rotation_axis)

# print(acc[:, -1])

new_acc = rotation_matrix @ acc

new_acc = new_acc - gravity_vector.T.reshape(3, 1)

print(new_acc[:, -1], rotation_angle * 180 / math.pi)

print(new_acc[0,:])