# code_generator/converters/tflite_parser/pack.py

from code_generator.operators.pack import PackOperator
from .utils import get_input_tensors, get_output_tensors, getTensorTypeStr
import tflite

def parse_pack(op, model):
    input_tensors = get_input_tensors(op, model)
    output_tensors = get_output_tensors(op, model)

    options = tflite.PackOptions()
    options.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)

    try:
        axis = options.Axis()
    except AttributeError:
        raise AttributeError("PackOptions object has no attribute 'Axis'. Ensure the tflite schema and flatbuffer are correctly set.")

    input_indices = [tensor.tensor_idx for tensor in input_tensors]
    output_idx = output_tensors[0].tensor_idx
    input_dtype = getTensorTypeStr(input_tensors[0].tensor.Type())
    output_dtype = getTensorTypeStr(output_tensors[0].tensor.Type())

    params = {
        "op": "PACK",
        "input_indices": input_indices,
        "output_idx": output_idx,
        "axis": axis,
        "input_dtype": input_dtype,
        "output_dtype": output_dtype,
    }

    for i, tensor in enumerate(input_tensors):
        params[f"input{i+1}_buf_add"] = str(tensor.tensor_idx)
        params[f"input{i+1}_buf_add_offset"] = "0"  # Adjust as needed

    return PackOperator(params)
