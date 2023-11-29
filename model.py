import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from torchsummary import summary
from thop import profile

SAMPLING_PERIOD = 3e-3
# THRESHOLD = 0.5

# WINDOW_LENGTH = 1.5
# STRIDE_LENGTH = 0.1

WINDOW_LENGTH = 0.3
STRIDE_LENGTH = 0.1

def proc_data(feat_x, feat_y, feat_z):
    # feat_z = np.sign(feat_z) * (feat_z ** 2) / (feat_x ** 2 + feat_y ** 2 + feat_z ** 2)
    return np.hstack((feat_x, feat_y, feat_z))

# class Scritch(nn.Module):
#   def __init__(self):
#     super(Scritch, self).__init__()

#     self.net = nn.Sequential(
#       nn.Linear(200, 400),
#       nn.ReLU(),
#       nn.Linear(400, 100),
#       nn.ReLU(),
#       nn.Linear(100, 1),
#       nn.Sigmoid()
#     )

#   def forward(self, x):
#     return self.net(x)

class ScritchData(Dataset):
    def __init__(self, filenames):
        arr = []
        for filename in filenames:
            arr.append(np.loadtxt(filename, delimiter=',', dtype=np.float32))
            # print(arr[-1].shape)
        arr = np.vstack(arr)
        # print(arr.shape)
        # z_data, label = arr[:, 2], arr[:, 3]
        x_data, y_data, z_data, label = arr.T

        window = lambda a, w, o: np.lib.stride_tricks.as_strided(a, strides = a.strides * 2, shape = (a.size - w + 1, w))[::o]
        window_size, sliding_size = int(WINDOW_LENGTH/SAMPLING_PERIOD), int(STRIDE_LENGTH/SAMPLING_PERIOD)

        feat_x = window(x_data, window_size, sliding_size)
        feat_y = window(y_data, window_size, sliding_size)
        feat_z = window(z_data, window_size, sliding_size)
        out_feat = np.sum(
            window(label, window_size, sliding_size),
            axis=1) > sliding_size//2
        
        # logistic regression
        # out_feat = out_feat.astype('float32')
        
        # one hot encoding
        # out_feat = F.one_hot(torch.from_numpy(out_feat.astype('float32')).long(), 2).float()\

        inp_feat = proc_data(feat_x, feat_y, feat_z)
        out_feat = np.eye(2)[out_feat.astype('int32')]

        self.inp_feat = inp_feat
        self.out_feat = out_feat
    
    def __len__(self):
        return len(self.out_feat)
    
    def __getitem__(self, idx):
        return self.inp_feat[idx], self.out_feat[idx]

# # logistic regression
# class Scritch(nn.Module):
#   def __init__(self):
#     super(Scritch, self).__init__()

#     self.net = nn.Sequential(
#       nn.Linear(int(WINDOW_LENGTH/SAMPLING_PERIOD), 400),
#       nn.ReLU(),
#       nn.Linear(400, 80),
#       nn.ReLU(),
#       nn.Linear(80, 16),
#       nn.ReLU(),
#       nn.Linear(16, 1),
#       nn.Sigmoid()
#     )

#   def forward(self, x):
#     return self.net(x)

# classification
# class Scritch(nn.Module):
#   def __init__(self):
#     super(Scritch, self).__init__()

#     self.net = nn.Sequential(
#       nn.Linear(int(WINDOW_LENGTH/SAMPLING_PERIOD), 400),
#       nn.ReLU(),
#       nn.Linear(400, 80),
#       nn.ReLU(),
#       nn.Linear(80, 16),
#       nn.ReLU(),
#       nn.Linear(16, 2),
#     )

#   def forward(self, x):
#     return self.net(x)

# class Scritch(nn.Module):
#   def __init__(self):
#     super(Scritch, self).__init__()

#     self.net = nn.Sequential(
#       nn.Linear(int(WINDOW_LENGTH/SAMPLING_PERIOD) * 3, 200),
#       nn.ReLU(),
#     #   nn.Dropout(0.2),
#       nn.Linear(200, 40),
#       nn.ReLU(),
#       nn.Linear(40, 2),
#     )

#   def forward(self, x):
#     return self.net(x)

class Scritch(nn.Module):
  def __init__(self):
    super(Scritch, self).__init__()

    in_feat = int(WINDOW_LENGTH/SAMPLING_PERIOD)
    net1_feat = 80
    net2_feat = 40

    self.net1_1 = nn.Sequential(
        nn.Linear(in_feat, net1_feat),
        nn.ReLU(),
    )
    self.net1_2 = nn.Sequential(
        nn.Linear(in_feat, net1_feat),
        nn.ReLU(),
    )
    self.net1_3 = nn.Sequential(
        nn.Linear(in_feat, net1_feat),
        nn.ReLU(),
    )

    self.net2 = nn.Sequential(
        nn.Linear(net1_feat * 3, net2_feat),
        nn.ReLU(),
        nn.Linear(net2_feat, 2)
    )

  def forward(self, x):
    # split (B, 300) into 3 * (B, 100)
    x, y, z = torch.chunk(x, 3, dim=1)
    x = self.net1_1(x)
    y = self.net1_2(y)
    z = self.net1_3(z)

    return self.net2(torch.cat((x, y, z), dim=1))
  
if __name__ == '__main__':
    # test dataset and print 
    dataset = ScritchData(['./data/data1.csv'])
    print(f'\nInput shape: {dataset[0][0].shape} Output shape: {dataset[0][1].shape}')
    
    in_shape = int(WINDOW_LENGTH/SAMPLING_PERIOD) * 3
    model = Scritch()
    summary(model, input_size=(in_shape,), device='cpu')
    macs, parm = profile(model, inputs=(torch.randn(1, in_shape), ))
    print(f'MACs: {int(macs)}, Params: {int(parm)}')