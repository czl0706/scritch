import torch
import torch.nn as nn
from torchsummary import summary
from thop import profile

### MODIFIABLE CONFIG ###

DATA_SAMPLING_PERIOD = 3e-3
MODL_SAMPLING_PERIOD = 1e-2 # 6e-3

# WINDOW_PERIOD = 0.45
# STRIDE_PERIOD = 0.1

WINDOW_PERIOD = 0.9
STRIDE_PERIOD = 0.1

### END OF MODIFIABLE CONFIG ###

WINDOW_LENGTH = int(WINDOW_PERIOD/MODL_SAMPLING_PERIOD)
STRIDE_LENGTH = int(STRIDE_PERIOD/MODL_SAMPLING_PERIOD)

DS_FACTOR = int(MODL_SAMPLING_PERIOD/DATA_SAMPLING_PERIOD)

assert WINDOW_LENGTH != 0
assert STRIDE_LENGTH != 0
assert DS_FACTOR != 0

# WINDOW_PERIOD = 1.5
# STRIDE_PERIOD = 0.1

# THRESHOLD = 0.5

# # logistic regression
# class Scritch(nn.Module):
#   def __init__(self):
#     super(Scritch, self).__init__()

#     self.net = nn.Sequential(
#       nn.Linear(WINDOW_LENGTH, 400),
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
#       nn.Linear(WINDOW_LENGTH, 400),
#       nn.ReLU(),
#       nn.Linear(400, 80),
#       nn.ReLU(),
#       nn.Linear(80, 16),
#       nn.ReLU(),
#       nn.Linear(16, 2),
#     )

#   def forward(self, x):
#     return self.net(x)

# 3-axis perception model
class Scritch(nn.Module):
  def __init__(self):
    super(Scritch, self).__init__()

    in_feat = WINDOW_LENGTH
    net1_feat = 20
    net2_feat = 40
    # net1_feat = 40
    # net2_feat = 60

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
        nn.Dropout(0.2),
        nn.Linear(net1_feat * 3, net2_feat),
        nn.ReLU(),
        nn.Linear(net2_feat, 2)
    )

  def forward(self, input):
    # split (B, C) into 3 * (B, C/3)
    x, y, z = torch.chunk(input, 3, dim=1)
    x = self.net1_1(x)
    y = self.net1_2(y)
    z = self.net1_3(z)

    return self.net2(torch.cat((x, y, z), dim=1))

class Scritch(nn.Module):
  def __init__(self):
    super(Scritch, self).__init__()

    in_feat = WINDOW_LENGTH // 2 * 4
    
    self.conv1 = nn.Conv1d(in_channels=3, out_channels=6, kernel_size=3, stride=2, padding=1, bias=True)
    self.conv2 = nn.Conv1d(in_channels=6, out_channels=4, kernel_size=3, stride=1, padding=1, bias=True)
    self.net1 = nn.Linear(in_feat, in_feat // 4)
    self.net2 = nn.Linear(in_feat // 4, 2)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(0.2)

  def forward(self, x):
    # (B, 3 * feat_size) -> (B, 3, feat_size)
    x = x.view(-1, 3, WINDOW_LENGTH)
    x = self.conv1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.relu(x)
    x = torch.flatten(x, start_dim=1)
    x = self.dropout(x)
    x = self.net1(x)  
    x = self.relu(x)  
    x = self.net2(x)    
    return x
  
if __name__ == '__main__':
    in_shape = WINDOW_LENGTH * 3
    model = Scritch()
    # Model summary and complexity
    summary(model, input_size=(in_shape,), device='cpu')
    macs, parm = profile(model, inputs=(torch.randn(1, in_shape), ))
    print(f'MACs: {int(macs)}, Params: {int(parm)}')
    print(f'DATA_SAMPLING_PERIOD: {DATA_SAMPLING_PERIOD}, MODL_SAMPLING_PERIOD: {MODL_SAMPLING_PERIOD}, WINDOW_PERIOD: {WINDOW_PERIOD}, STRIDE_PERIOD: {STRIDE_PERIOD}')
    print(f'WINDOW_LENGTH: {WINDOW_LENGTH}, STRIDE_LENGTH: {STRIDE_LENGTH}, DS_FACTOR: {DS_FACTOR}')