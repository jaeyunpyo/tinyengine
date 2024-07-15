# code_generator/operators/pack.py

import warnings
from .basic_utils import basicOperator, deep_copy_dicts, overwrite_dicts

default_params = {
    "op": "PACK",
    "input_indices": None,
    "output_idx": None,
    "axis": None,
    "input_dtype": None,
    "output_dtype": None,
}

class PackOperator(basicOperator):
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

    def generate_inference_str(self):
        params = self.params
        input_buffers = []

        for i in range(len(params["input_indices"])):
            input_add = str(params[f"input{i+1}_buf_add"])
            input_offset = str(params[f"input{i+1}_buf_add_offset"])
            input_buffers.append(
                self._getBufferstrCast(input_add, input_offset, dtype=params["input_dtype"])
            )

        input_str = ", ".join(input_buffers)
        output_str = self._getBufferstrCast(str(params["output_buf_add"]), str(params["output_buf_add_offset"]), dtype=params["output_dtype"])
        
        return f"pack({input_str}, {output_str}, {len(params['input_indices'])}, {params['axis']});"
