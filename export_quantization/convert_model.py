from get_model_and_params import *
import torch
import onnx


save_path = '../models/Scritch.onnx'

torch_model = Scritch()
torch_input = torch.randn(1, int(WINDOW_LENGTH/SAMPLING_PERIOD) * 3)

# https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
torch.onnx.export(torch_model,               # model being run
                  torch_input,                         # model input (or a tuple for multiple inputs)
                  save_path,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

onnx_model = onnx.load(save_path)
onnx.checker.check_model(onnx_model)