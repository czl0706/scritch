#! /bin/bash

target_chip="esp32"

# https://docs.espressif.com/projects/esp-dl/zh_CN/latest/esp32/tutorials/deploying-models-through-tvm.html#
python -m onnxruntime.quantization.preprocess --input model.onnx --output model_opt.onnx
python esp_quantize_onnx.py --input_model model_opt.onnx --output_model model_quant.onnx --calibrate_dataset cal_ds.npy
python export_onnx_model.py --target_chip $target_chip --model_path model_quant.onnx --img_path input_sample.npy --template_path "./template_project_for_model" --out_path "./scritch"
