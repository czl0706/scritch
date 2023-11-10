import torch.nn as nn

SAMPLING_PERIOD = 5e-3
WINDOW_LENGTH = 1.5
STRIDE_LENGTH = 0.1

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