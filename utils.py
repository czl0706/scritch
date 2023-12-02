import torch
from torch.utils.data import Dataset
import numpy as np

from model import *

ds_list = ['./data/data1.csv', 
           './data/data2.csv',
           './data/data3.csv',
           './data/data4.csv',
           './data/data5.csv',
           './data/data6.csv']

def proc_data(feat_x, feat_y, feat_z):
    # feat_z = np.sign(feat_z) * (feat_z ** 2) / (feat_x ** 2 + feat_y ** 2 + feat_z ** 2)
    return np.hstack((feat_x, feat_y, feat_z))

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
        
        x_data, y_data, z_data, label = arr #.T

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
    print(f'Number of training data: {len(dataset)}')
    print(f'Input shape: {dataset[0][0].shape} Output shape: {dataset[0][1].shape}')

    # test the shape of data is fit to model
    model = Scritch()
    model(torch.from_numpy(dataset[0][0]).unsqueeze(0))