import torch
from get_model_and_params import *
import numpy as np

def export_params_to_header(model, filename):
    """
    Export PyTorch model parameters to a C header file.

    Args:
        model (torch.nn.Module): PyTorch model.
        filename (str): Name of the output header file.
    """
    params = {}

    # Collect model parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            params[name] = param.data.cpu().numpy()
            print(name, param.data.cpu().numpy().shape)

    # Export params to C header file
    with open(filename, 'w') as f:
        for name, param in params.items():
            write_parameter_to_header(f, name, param)

def write_parameter_to_header(file, name, param):
    """
    Write a parameter to a C header file.

    Args:
        file: File object for writing.
        name (str): Name of the parameter.
        param (numpy.ndarray): Parameter data.
    """
    # Write parameter to header file based on its shape
    shape_str = ']['.join(map(str, param.shape))
    file.write('const float {}[{}] = '.format(name.replace('.', '_'), shape_str))
    file.write('{')

    if len(param.shape) == 1:
        write_array_elements(file, param)
    elif len(param.shape) > 1:
        for i in range(param.shape[0]):
            file.write('{')
            write_array_elements(file, param[i])
            file.write('}')
            if i != param.shape[0] - 1:
                file.write(', ')

    file.write('};\n')

def write_array_elements(file, array):
    """
    Write array elements to the header file.

    Args:
        file: File object for writing.
        array (numpy.ndarray): Array data.
    """
    for i, value in enumerate(array):
        if isinstance(value, (list, np.ndarray)):
            write_array_elements(file, value)
        else:
            file.write(str(value))

        if i != len(array) - 1:
            file.write(', ')

# Example usage
model = Scritch()
model.load_state_dict(torch.load(torch_model_path))
model.eval()

export_params_to_header(model, 'model.h')
