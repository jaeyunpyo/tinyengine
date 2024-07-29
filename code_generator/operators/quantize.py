# code_generator/operators/quantize.py
import warnings
from .basic_utils import basicOperator, deep_copy_dicts, overwrite_dicts

default_params = {
    "op": "QUANTIZE",
    "input_idx": None,
    "output_idx": None,
    "scale": None,
    "zero_point": None,
    "input_dtype": None,
    "output_dtype": None,
    "input1_buf_add": None,
    "input1_buf_add_offset": None,
    "output_buf_add": None,
    "output_buf_add_offset": None,
    "input_buffer_size": None,
}

class QuantizeOperator(basicOperator):
    def __init__(self, params: dict) -> None:
        self.params = deep_copy_dicts(default_params)
        overwrite_dicts(self.params, params)
        super().__init__()

        if self.params["input_idx"] is not None:
            self._add_input(self.params["input_idx"], self.params["input_dtype"])

        if self.params["output_idx"] is not None:
            self._add_output(self.params["output_idx"], self.params["output_dtype"])

        if None in [self.params[key] for key in self.params if key != "axis"]:
            warnings.warn(f"Parameters are not all set for op {self.params['op']}")

    def generate_inference_str(self):
        params = self.params
        input_add = str(params["output_buf_add"])
        input_offset = str(params["output_buf_add_offset"])
        output_add = str(params["output_buf_add"])
        output_offset = str(params["output_buf_add_offset"])
        input_str = self._getBufferstrCast(input_add, input_offset, dtype=params["input_dtype"])
        output_str = self._getBufferstrCast(output_add, output_offset, dtype=params["output_dtype"])

        scale_str = str(params["scale"])
        zero_point_str = str(params["zero_point"])
        input_buffer_size_str = str(params["input_buffer_size"])

        return f"quantize({input_str}, {output_str}, {scale_str}, {zero_point_str}, {input_buffer_size_str} );"

