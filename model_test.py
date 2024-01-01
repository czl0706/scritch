import numpy as np
import torch
from utils import *

model = Scritch() #.cuda()
model.load_state_dict(torch.load('./models/model.pt'))

model.eval()

np.random.seed(42)
inp_feat = [0.37454012, 0.9507143, 0.7319939, 0.5986585, 0.15601864, 0.15599452, 0.058083612, 0.8661761, 0.601115, 0.7080726, 0.020584494, 0.96990985, 0.83244264, 0.21233912, 0.18182497, 0.1834045, 0.30424225, 0.52475643, 0.43194503, 0.29122913, 0.6118529, 0.13949387, 0.29214466, 0.36636186, 0.45606998, 0.785176, 0.19967379, 0.5142344, 0.59241456, 0.046450414, 0.60754484, 0.17052412, 0.06505159, 0.94888556, 0.965632, 0.80839735, 0.30461377, 0.09767211, 0.684233, 0.4401525, 0.12203824, 0.4951769, 0.03438852, 0.9093204, 0.25877997, 0.66252226, 0.31171107, 0.52006805, 0.54671025, 0.18485446, 0.96958464, 0.77513283, 0.93949896, 0.89482737, 0.5979, 0.9218742, 0.088492505, 0.19598286, 0.04522729, 0.32533032, 0.3886773, 0.27134904, 0.8287375, 0.35675332, 0.2809345, 0.54269606, 0.14092423, 0.802197, 0.07455064, 0.9868869, 0.77224475, 0.19871569, 0.005522117, 0.81546146, 0.7068573, 0.7290072, 0.77127033, 0.07404465, 0.35846573, 0.11586906, 0.86310345, 0.6232981, 0.33089802, 0.06355835, 0.31098232, 0.32518333, 0.72960615, 0.63755745, 0.88721275, 0.47221494, 0.119594246, 0.7132448, 0.76078504, 0.5612772, 0.7709672, 0.4937956, 0.52273285, 0.42754102, 0.025419127, 0.107891425, 0.031429186, 0.6364104, 0.31435597, 0.5085707, 0.9075665, 0.24929222, 0.41038293, 0.75555116, 0.22879817, 0.07697991, 0.28975144, 0.16122128, 0.92969763, 0.80812037, 0.6334038, 0.8714606, 0.8036721, 0.18657006, 0.892559, 0.5393422, 0.80744016, 0.8960913, 0.31800348, 0.11005192, 0.22793517, 0.42710778, 0.81801474, 0.8607306, 0.0069521307, 0.5107473, 0.417411, 0.22210781, 0.119865365, 0.33761516, 0.9429097, 0.32320294, 0.5187906, 0.70301896, 0.3636296, 0.9717821, 0.9624473, 0.2517823, 0.4972485, 0.30087832, 0.2848405, 0.03688695, 0.6095643, 0.50267905, 0.05147875, 0.27864647]
inp_feat = np.array(inp_feat).reshape(1, 150).astype('float32')

print(model(torch.from_numpy(inp_feat)))