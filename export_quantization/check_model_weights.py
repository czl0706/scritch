from get_model_and_params import *
import torch
import numpy as np
import matplotlib.pyplot as plt

model = Scritch()
model.load_state_dict(torch.load(torch_model_path))
model.eval()

weights = []

for param in model.parameters():
    weights.append(np.hstack(param.data.cpu().numpy()))

weights = np.concatenate(weights)

plt.hist(weights, bins=50, alpha=0.75, color='b', edgecolor='black')

plt.title('Histogram of Model Weights')
plt.xlabel('Weight Value')
plt.ylabel('Frequency')

plt.show()
