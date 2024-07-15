# code_generator/operators/batch_matmul.py

import warnings
from .basic_utils import basicOperator, deep_copy_dicts, overwrite_dicts

__all__ = ["BatchMatMulOperator"]

default_params = {
    # Operator related
    "op": "BATCH_MATMUL",
    # Tensor related
    "input_idx": None,
    "input2_idx": None,
    "output_idx": None,
    "input_shape": None,
    "input2_shape": None,
    "output_shape": None,
    "input_dtype": "float32",
    "input2_dtype": "float32",
    "output_dtype": "float32",
    # Quantization parameters
    "input_zero_point": None,
    "input2_zero_point": None,
    "output_zero_point": None,
    "input_scale": None,
    "input2_scale": None,
    "output_scale": None,
    "input_multiplier": None,
    "input2_multiplier": None,
    "output_multiplier": None,
    "input_shift": None,
    "input2_shift": None,
    "output_shift": None,
    # Additional parameters
    "adj_x": False,
    "adj_y": False,
}

class BatchMatMulOperator(basicOperator):
    def __init__(self, params: dict) -> None:
        self.params = deep_copy_dicts(default_params)
        overwrite_dicts(self.params, params)
        super().__init__()
        
        # Handle input/output tensors
        self._add_input(
            self.params["input_idx"],
            self.params["input_dtype"],
            *self.params["input_shape"]
        )
        self._add_input(
            self.params["input2_idx"],
            self.params["input2_dtype"],
            *self.params["input2_shape"]
        )
        self._add_output(
            self.params["output_idx"],
            self.params["output_dtype"],
            *self.params["output_shape"]
        )

        if None in self.params.values():
            warnings.warn(f"parameters are not all set for op {self.params['op']}")
            
    def _get_param_value(self, param_name, default_value):
        return self.params[param_name] if self.params[param_name] is not None else default_value

    def generate_inference_str(self):
        input_str = self._getBufferstrCast(
            self.params['input1_buf_add'], self.params['input1_buf_add_offset'], dtype=self.params["input_dtype"]
        )
        input2_str = self._getBufferstrCast(
            self.params['input2_buf_add'], self.params['input2_buf_add_offset'], dtype=self.params["input2_dtype"]
        )
        output_str = self._getBufferstrCast(
            self.params['output_buf_add'], self.params['output_buf_add_offset'], dtype=self.params["output_dtype"]
        )

        # Convert boolean parameters to C++-style true/false
        adj_x = 'true' if self.params['adj_x'] else 'false'
        adj_y = 'true' if self.params['adj_y'] else 'false'
        
        input_zero_point = self._get_param_value('input_zero_point', 0)
        input2_zero_point = self._get_param_value('input2_zero_point', 0)
        output_zero_point = self._get_param_value('output_zero_point', 0)
        input_scale = self._get_param_value('input_scale', 1.0)
        input2_scale = self._get_param_value('input2_scale', 1.0)
        output_scale = self._get_param_value('output_scale', 1.0)
        input_shift = self._get_param_value('input_shift', 0)
        input2_shift = self._get_param_value('input2_shift', 0)
        output_shift = self._get_param_value('output_shift', 0)
        input_multiplier = self._get_param_value('input_multiplier', 0)
        input2_multiplier = self._get_param_value('input2_multiplier', 0)
        output_multiplier = self._get_param_value('output_multiplier', 0)


        string = (
            f"batch_matmul({input_str}, {input2_str}, {output_str}, "
            f"{self.params['input_shape'][0]}, {self.params['input_shape'][1]}, {self.params['input_shape'][2]}, "
            f"{self.params['input2_shape'][1]}, "
            f"{adj_x}, {adj_y}, "
            f"{input_zero_point}, {input2_zero_point}, {output_zero_point}, "
            f"{input_scale}, {input2_scale}, {output_scale}, "
            f"{input_shift}, {input2_shift}, {output_shift}, "
            f"{input_multiplier}, {input2_multiplier}, {output_multiplier});\n"
        )

        return string
