import torch
from torch.utils.data import Dataset
import numpy as np
from io import StringIO

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

ds_list = [
            './data/data_1.txt',
            './data/data_2.txt',
            # './data/data_3.txt',
           ]

lbl_list = [
            './data/label_1.txt',
            './data/label_2.txt',
            # './data/label_3.txt',
            ] 

# def proc_data(feat_x, feat_y, feat_z):
#     return np.hstack((feat_x, feat_y, feat_z))

class ScritchData(Dataset):
    def __init__(self, ds_list, lbl_list):
        x = []
        res = []
        for filename in ds_list:
            with open(filename, 'r') as f:
                x = f.readlines()
                
                size = len(x)
                idx_list = [idx + 1 for idx, val in enumerate(x) if val == '\n']
                tmp = [x[i: j] for i, j in
                    zip([0] + idx_list, idx_list +
                        ([size] if idx_list[-1] != size else []))]
                
                tmp = [''.join(x) for x in tmp]
                
                # load tmp into numpy array
                res += [np.genfromtxt(StringIO(x), delimiter=",") for x in tmp]
        
        # (1280, 50, 3) -> (1280, 3, 50)
        res = np.array(res)
        res = np.transpose(res, (0, 2, 1))
        
        label = [np.loadtxt(x, dtype=np.float32) for x in lbl_list]
        label = np.hstack(label)
        label = label > 0.5
        
        label = np.eye(2)[label.astype('int32')]
        
        # print(res.shape)
        # print(label.shape)

        self.inp_feat = res.astype('float32')
        self.out_feat = label.astype('float32')
    
    def __len__(self):
        return len(self.out_feat)
    
    def __getitem__(self, idx):
        return self.inp_feat[idx], self.out_feat[idx]
    
def get_dataset() -> ScritchData:
    return ScritchData(ds_list, lbl_list)
    
if __name__ == '__main__':
    dataset = get_dataset()
    labels = [x for _, x in dataset]
    neg, pos = np.sum(labels, axis=0).astype('int32')
    
    print(f'Number of training data: {len(dataset)}')
    print(f'Positive samples: {pos}, Negative samples: {neg}')  
    print(f'Input shape: {dataset[0][0].shape} Output shape: {dataset[0][1].shape}')
    
    # test the shape of data is fit to model
    model = Scritch()
    model(torch.from_numpy(dataset[0][0]).unsqueeze(0))