import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

# SAMPLING_PERIOD = 3e-3
# WINDOW_LENGTH = 1.5
# STRIDE_LENGTH = 0.1
# THRESHOLD = 0.5

SAMPLING_PERIOD = 3e-3
WINDOW_LENGTH = 0.5
STRIDE_LENGTH = 0.1
THRESHOLD = 0.5

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
        z_data, label = arr[:, 2], arr[:, 3]
        # x_data, y_data, z_data, label = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

        window = lambda a, w, o: np.lib.stride_tricks.as_strided(a, strides = a.strides * 2, shape = (a.size - w + 1, w))[::o]
        window_size, sliding_size = int(WINDOW_LENGTH/SAMPLING_PERIOD), int(STRIDE_LENGTH/SAMPLING_PERIOD)

        inp_feat = window(z_data, window_size, sliding_size)
        out_feat = np.sum(
            window(label, window_size, sliding_size),
            axis=1) > sliding_size//2
        out_feat = out_feat.astype('float32')

        self.inp_feat = inp_feat
        self.out_feat = out_feat
    
    def __len__(self):
        return len(self.out_feat)
    
    def __getitem__(self, idx):
        return self.inp_feat[idx], self.out_feat[idx]

class Scritch(nn.Module):
  def __init__(self):
    super(Scritch, self).__init__()

    self.net = nn.Sequential(
      nn.Linear(int(WINDOW_LENGTH/SAMPLING_PERIOD), 400),
      nn.ReLU(),
      nn.Linear(400, 80),
      nn.ReLU(),
      nn.Linear(80, 16),
      nn.ReLU(),
      nn.Linear(16, 1),
      nn.Sigmoid()
    )

  def forward(self, x):
    return self.net(x)