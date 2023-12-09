import numpy as np 

DATA_SAMPLING_PERIOD = 3e-3
MODL_SAMPLING_PERIOD = 1e-2 # 6e-3

WINDOW_PERIOD = 0.45
STRIDE_PERIOD = 0.1

### END OF MODIFIABLE CONFIG ###

WINDOW_LENGTH = int(WINDOW_PERIOD/MODL_SAMPLING_PERIOD)
STRIDE_LENGTH = int(STRIDE_PERIOD/MODL_SAMPLING_PERIOD)

DS_FACTOR = int(MODL_SAMPLING_PERIOD/DATA_SAMPLING_PERIOD)

ds_list = ['./data/data1.csv', 
           './data/data2.csv',
           './data/data3.csv',
           './data/data4.csv',
           './data/data5.csv',
           './data/data6.csv',
           './data/data7.csv']

arr = []
for filename in ds_list:
    arr.append(np.loadtxt(filename, delimiter=',', dtype=np.float32))
    # print(arr[-1].shape)
arr = np.vstack(arr)

print(arr.shape)

# sampling
arr = arr.T[::, ::DS_FACTOR]

# print(arr.shape)

x_data, y_data, z_data, label = arr #.T






window = lambda m: np.lib.stride_tricks.as_strided(m, 
                                                    strides = m.strides * 2, 
                                                    shape = (m.size - WINDOW_LENGTH + 1, WINDOW_LENGTH)
                                                    )[::STRIDE_LENGTH]

feat_x, feat_y, feat_z = window(x_data), window(y_data), window(z_data)
out_feat = np.sum(window(label), axis=1) > STRIDE_LENGTH//2

# inp_feat = np.hstack((feat_x, feat_y, feat_z))
out_feat = np.eye(2)[out_feat.astype('int32')]

print(WINDOW_LENGTH, STRIDE_LENGTH)
print(feat_x.shape)
print(out_feat.shape)