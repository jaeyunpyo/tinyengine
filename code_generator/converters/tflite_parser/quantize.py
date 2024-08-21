# code_generator/converters/tflite_parser/quantize.py

from code_generator.operators.quantize import QuantizeOperator
from .utils import get_input_tensors, get_output_tensors, getTensorTypeStr
from code_generator.tflite.QuantizeOptions import QuantizeOptions
import numpy as np

def calculate_buffer_size(tensor):
    shape = tensor.tensor.ShapeAsNumpy()
    dtype = getTensorTypeStr(tensor.tensor.Type())
    
    # Calculate the buffer size
    buffer_size = np.prod(shape)    
    
    if dtype == "float32":
        buffer_size *= 4  # 4 bytes for float32
    elif dtype == "int32":
        buffer_size *= 4  # 4 bytes for int32
    elif dtype == "int8":
        buffer_size *= 1  # 1 byte for int8
    elif dtype == "uint8":
        buffer_size *= 1  # 1 byte for uint8
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    return buffer_size

def parse_quantize(op, model):
    # Get input and output tensors
    input_tensors = get_input_tensors(op, model)
    output_tensors = get_output_tensors(op, model)

    if len(input_tensors) != 1 or len(output_tensors) != 1:
        raise ValueError("QUANTIZE operator should have one input and one output tensor")

    input_tensor = input_tensors[0]
    output_tensor = output_tensors[0]

    # Get quantization options if available
    options = QuantizeOptions()
    if op.BuiltinOptions() is not None:
        options.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)


    # Get quantization parameters
    input_quant_params = input_tensor.qnn_params
    output_quant_params = output_tensor.qnn_params

    if input_quant_params is None or output_quant_params is None:
        raise ValueError("Input or output tensor does not have quantization parameters")
    
    # Calculate input buffer size
    input_buffer_size = calculate_buffer_size(input_tensor)

    params = {
        "op": "QUANTIZE",
        "input_idx" : input_tensor.tensor_idx,
        "output_idx" : output_tensor.tensor_idx,
        "input_dtype": getTensorTypeStr(input_tensor.tensor.Type()),
        "output_dtype": getTensorTypeStr(output_tensor.tensor.Type()),        
        "scale": input_quant_params["scale"],
        "zero_point": input_quant_params["zero_point"],
        "input1_buf_add": "input_buffer",
        "input1_buf_add_offset": "input_offset",
        "output_buf_add": "output_buffer",
        "output_buf_add_offset": "output_offset",
        "input_buffer_size": input_buffer_size,  # Add input buffer size to params
    }

    return QuantizeOperator(params)