from get_model_and_params import *
from onnx import opti

# https://docs.espressif.com/projects/esp-dl/zh_CN/latest/esp32/tutorials/deploying-models-through-tvm.html#
# https://docs.espressif.com/projects/esp-dl/zh_CN/latest/esp32/tutorials/deploying-models.html

# Optimize the onnx model
# optimized_model_path = optimize_fp_model(onnx_save_path)