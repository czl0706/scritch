import os
import sys
sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 '../'))

from model import *

torch_model_path = '../models/model.pt'
onnx_save_path = '../models/Scritch.onnx'