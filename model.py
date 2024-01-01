import torch
import torch.nn as nn
from torchsummary import summary
from thop import profile

### MODIFIABLE CONFIG ###

DATA_SAMPLING_PERIOD = 1e-2
MODL_SAMPLING_PERIOD = 1e-2

WINDOW_PERIOD = 0.5
STRIDE_PERIOD = 0.25

### MODIFIABLE CONFIG ###

WINDOW_LENGTH = int(WINDOW_PERIOD/MODL_SAMPLING_PERIOD)
STRIDE_LENGTH = int(STRIDE_PERIOD/MODL_SAMPLING_PERIOD)

DS_FACTOR = int(MODL_SAMPLING_PERIOD/DATA_SAMPLING_PERIOD)

assert WINDOW_LENGTH != 0
assert STRIDE_LENGTH != 0
assert DS_FACTOR != 0
  
class Scritch(nn.Module):
  def __init__(self):
    super(Scritch, self).__init__()

    self.conv1 = nn.Conv1d(in_channels=3, out_channels=6, kernel_size=5, stride=2, padding=2, bias=True)
    self.net1 = nn.Linear(150, 30)
    self.net2 = nn.Linear(30, 2)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(0.3)

  def forward(self, x):
    # (B, 3 * feat_size) -> (B, 3, feat_size)
    x = x.view(-1, 3, WINDOW_LENGTH)
    x = self.conv1(x)
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