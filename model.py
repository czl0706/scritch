import torch
import torch.nn as nn
from torchsummary import summary
from thop import profile

DATA_SAMPLING_PERIOD = 3e-3
MODEL_INPUT_PERIOD   = 1e-2 # 6e-3

WINDOW_PERIOD = 0.45
STRIDE_PERIOD = 0.1

WINDOW_LENGTH = int(WINDOW_PERIOD/MODEL_INPUT_PERIOD)
STRIDE_LENGTH = int(STRIDE_PERIOD/MODEL_INPUT_PERIOD)

DS_FACTOR = int(MODEL_INPUT_PERIOD/DATA_SAMPLING_PERIOD)

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
    net1_feat = 60
    net2_feat = 30

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

  def forward(self, x):
    # split (B, 300) into 3 * (B, 100)
    x, y, z = torch.chunk(x, 3, dim=1)
    x = self.net1_1(x)
    y = self.net1_2(y)
    z = self.net1_3(z)

    return self.net2(torch.cat((x, y, z), dim=1))

# class Scritch(nn.Module):
#   def __init__(self):
#     super(Scritch, self).__init__()

#     in_feat = WINDOW_LENGTH
#     net1_feat = 40
#     net2_feat = 80

#     self.net1_1 = nn.Sequential(
#         nn.Linear(in_feat, net1_feat),
#         nn.ReLU(),
#     )
#     self.net1_2 = nn.Sequential(
#         nn.Linear(in_feat, net1_feat),
#         nn.ReLU(),
#     )
#     self.net1_3 = nn.Sequential(
#         nn.Linear(in_feat, net1_feat),
#         nn.ReLU(),
#     )

#     self.net2 = nn.Sequential(
#         nn.Linear(net1_feat * 3, net2_feat),
#         nn.ReLU(),
#         nn.Linear(net2_feat, 2)
#     )

#   def forward(self, x):
#     # split (B, 300) into 3 * (B, 100)
#     x, y, z = torch.chunk(x, 3, dim=1)
#     x = self.net1_1(x)
#     y = self.net1_2(y)
#     z = self.net1_3(z)

#     return self.net2(torch.cat((x, y, z), dim=1))
  
if __name__ == '__main__':
    in_shape = WINDOW_LENGTH * 3
    model = Scritch()
    # Model summary and complexity
    summary(model, input_size=(in_shape,), device='cpu')
    macs, parm = profile(model, inputs=(torch.randn(1, in_shape), ))
    print(f'MACs: {int(macs)}, Params: {int(parm)}')
    print(f'WINDOW_LENGTH: {WINDOW_LENGTH}, STRIDE_LENGTH: {STRIDE_LENGTH}, DS_FACTOR: {DS_FACTOR}')