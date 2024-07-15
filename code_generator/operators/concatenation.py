# code_generator/operators/concatenation.py

import warnings
from .basic_utils import basicOperator, deep_copy_dicts, overwrite_dicts

default_params = {
    "op": "CONCATENATION",
    "input_indices": None,
    "input_shapes": None,
    "output_idx": None,
    "axis": None,
    "input_dtype": None,
    "output_dtype": None,
}

class ConcatenationOperator(basicOperator):
    def __init__(self, params: dict) -> None:
        self.params = deep_copy_dicts(default_params)
        overwrite_dicts(self.params, params)
        super().__init__()

        if self.params["input_indices"] is not None:
            for idx in self.params["input_indices"]:
                self._add_input(idx, self.params["input_dtype"])

        if self.params["output_idx"] is not None:
            self._add_output(self.params["output_idx"], self.params["output_dtype"])

        if None in default_params.values():
            warnings.warn(f"parameters are not all set for op {self.params['op']}")
            
    def _get_input_size(self, shape):
        # Calculate the size of the input tensor based on its shape
        size = 1
        for dim in shape:
            size *= dim
        return size

    def generate_inference_str(self):
        params = self.params
        input_sizes = [self._get_input_size(shape) for shape in params["input_shapes"]]        
        input_sizes_strs = ", ".join([str(size) for size in input_sizes])
        
        input_str = ", ".join([
            self._getBufferstrCast(params[f"input1_buf_add"], 
                                params[f"input1_buf_add_offset"], 
                                dtype=params["input_dtype"]) 
            , 
            self._getBufferstrCast(params[f"input2_buf_add"], 
                                    params[f"input2_buf_add_offset"], 
                                    dtype=params["input_dtype"]) 
        ])
        output_str = self._getBufferstrCast(params["output_buf_add"], params["output_buf_add_offset"], dtype=params["output_dtype"])
        
        string = (
            f"concatenate({input_str}, {input_sizes_strs}, {output_str}, {params['axis']});\n"
        )

        return string
