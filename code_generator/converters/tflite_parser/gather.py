import math

import numpy as np
from code_generator.operators.gather import GatherOperator
from code_generator.tflite import Model
from code_generator.tflite.BuiltinOperator import BuiltinOperator
from code_generator.tflite.TensorType import TensorType

from .utils import (
    get_input_tensors,
    get_output_tensors,
    getTensorTypeStr,
)
def parse_gather(op, model):
    input_tensors = get_input_tensors(op, model)
    output_tensors = get_output_tensors(op, model)
    
    indices_idx = None
    indices_shape = [1]
    indices_dtype = "int32"

    if len(input_tensors) >= 2:
        indices_idx = input_tensors[1].tensor_idx
        indices_shape = input_tensors[1].tensor.ShapeAsNumpy()
        indices_dtype = getTensorTypeStr(input_tensors[1].tensor.Type())

    params = {
        "op": "GATHER",
        "input_idx": input_tensors[0].tensor_idx,
        "indices_idx": indices_idx,
        "output_idx": output_tensors[0].tensor_idx,
        "input_dtype": getTensorTypeStr(input_tensors[0].tensor.Type()),
        "indices_dtype": indices_dtype,
        "output_dtype": getTensorTypeStr(output_tensors[0].tensor.Type()),
        "input_shape": input_tensors[0].tensor.ShapeAsNumpy(),
        "indices_shape": indices_shape,
        "output_shape": output_tensors[0].tensor.ShapeAsNumpy()
    }
    return GatherOperator(params)
    