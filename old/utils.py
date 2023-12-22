import torch
from torch.utils.data import Dataset
import numpy as np

from model import *

# ds_list = ['./data/data1.csv', 
#            './data/data2.csv',
#            './data/data3.csv',
#            './data/data4.csv',
#            './data/data5.csv',
#            './data/data6.csv',
#            './data/data7.csv',
#         #    './data/data8.csv',
#            './data/data9.csv',
#            ]

ds_list = ['./data/data.csv']

from scipy.spatial.transform import Rotation as R
def proc_data(feat_x, feat_y, feat_z):
    # feat_z = np.sign(feat_z) * (feat_z ** 2) / (feat_x ** 2 + feat_y ** 2 + feat_z ** 2)
    
    # feat = np.hstack((feat_x, feat_y, feat_z))
    # feat should be (batch_size, window_size, 3) 
    feat = np.stack((feat_x, feat_y, feat_z), axis=2)
    
    acceleration_vector = np.mean(feat_x, axis=1), np.mean(feat_y, axis=1) , np.mean(feat_z, axis=1)
    acceleration_vector = np.array(acceleration_vector).T
    # print(feat.shape, acceleration_vector.shape)
    
    gravity_vector = np.array([0, 0, 1])
    
    # 計算旋轉軸
    rotation_axis = np.cross(acceleration_vector, gravity_vector)
    rotation_axis /= np.linalg.norm(rotation_axis)
    
    # 計算旋轉角度
    rotation_angle = np.arccos(np.dot(acceleration_vector, gravity_vector))

    rotation_angle = rotation_angle.reshape(-1, 1)
    
    # 構造旋轉矩陣
    rotation_matrix = R.from_rotvec(rotation_angle * rotation_axis).as_matrix()
    
    print(rotation_matrix.shape, feat.shape)
    
    new_acc = np.einsum('bij,bkj->bik', rotation_matrix, feat)
    
    print(new_acc.shape)

    new_acc = new_acc - gravity_vector.T.reshape(3, 1)
    
    print(new_acc)
    
    # (2405, 3, 60) -> (2405, 180)
    new_acc = new_acc.reshape(new_acc.shape[0], -1)
    
    return new_acc.astype('float32')
    # return np.hstack((feat_x, feat_y, feat_z))

class ScritchData(Dataset):
    def __init__(self, filenames):
        arr = []
        for filename in filenames:
            arr.append(np.loadtxt(filename, delimiter=',', dtype=np.float32))
            # print(arr[-1].shape)
        arr = np.vstack(arr)
        # print(arr.shape)
        # z_data, label = arr[:, 2], arr[:, 3]
        
        # sampling
        arr = arr.T[::, ::DS_FACTOR]
        
        # print(arr.shape)
        x_data, y_data, z_data = arr[1:4, :]
        label = arr[7, :] #.T

        window = lambda m: np.lib.stride_tricks.as_strided(m, 
                                                           strides = m.strides * 2, 
                                                           shape = (m.size - WINDOW_LENGTH + 1, WINDOW_LENGTH)
                                                           )[::STRIDE_LENGTH]

        feat_x, feat_y, feat_z = window(x_data), window(y_data), window(z_data)
        out_feat = np.sum(window(label), axis=1) > STRIDE_LENGTH//2
        
        # window = lambda m, w, o: np.lib.stride_tricks.as_strided(m, strides = m.strides * 2, shape = (m.size - w + 1, w))[::o]
        # window_size, sliding_size = WINDOW_LENGTH, STRIDE_LENGTH

        # feat_x = window(x_data)
        # feat_y = window(y_data)
        # feat_z = window(z_data)
        # out_feat = np.sum(window(label, window_size, sliding_size),
        #                   axis=1) > sliding_size//2
        
        # logistic regression
        # out_feat = out_feat.astype('float32')
        
        # one hot encoding
        # out_feat = F.one_hot(torch.from_numpy(out_feat.astype('float32')).long(), 2).float()

        inp_feat = proc_data(feat_x, feat_y, feat_z)
        out_feat = np.eye(2)[out_feat.astype('int32')]

        self.inp_feat = inp_feat
        self.out_feat = out_feat
    
    def __len__(self):
        return len(self.out_feat)
    
    def __getitem__(self, idx):
        return self.inp_feat[idx], self.out_feat[idx]
    
def get_dataset() -> ScritchData:
    return ScritchData(ds_list)
    
if __name__ == '__main__':
    dataset = get_dataset()
    labels = [x for _, x in dataset]
    pos, neg = np.sum(labels, axis=0).astype('int32')
    
    print(f'Number of training data: {len(dataset)}')
    print(f'Positive samples: {pos}, Negative samples: {neg}')  
    print(f'Input shape: {dataset[0][0].shape} Output shape: {dataset[0][1].shape}')
    
    # test the shape of data is fit to model
    model = Scritch()
    model(torch.from_numpy(dataset[0][0]).unsqueeze(0))