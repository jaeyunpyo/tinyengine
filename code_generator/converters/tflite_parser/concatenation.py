# code_generator/converters/tflite_parser/concatenation.py

from code_generator.operators.concatenation import ConcatenationOperator
from .utils import get_input_tensors, get_output_tensors, getTensorTypeStr
from tflite.ConcatenationOptions import ConcatenationOptions

def parse_concatenation(op, model):
    input_tensors = get_input_tensors(op, model)
    output_tensors = get_output_tensors(op, model)

    options = ConcatenationOptions()
    options.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)

    try:
        axis = options.Axis()
    except AttributeError:
        raise AttributeError("ConcatenationOptions object has no attribute 'Axis'. Ensure the tflite schema and flatbuffer are correctly set.")

    input_indices = [tensor.tensor_idx for tensor in input_tensors]
    input_shapes = [tensor.tensor.ShapeAsNumpy() for tensor in input_tensors]
    output_idx = output_tensors[0].tensor_idx
    input_dtype = getTensorTypeStr(input_tensors[0].tensor.Type())
    output_dtype = getTensorTypeStr(output_tensors[0].tensor.Type())

    params = {
        "op": "CONCATENATION",
        "input_indices": input_indices,
        "input_shapes": input_shapes,
        "output_idx": output_idx,
        "axis": axis,
        "input_dtype": input_dtype,
        "output_dtype": output_dtype,
    }

    return ConcatenationOperator(params)
