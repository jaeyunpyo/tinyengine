# code_generator/operators/gather.py

import warnings
from .basic_utils import basicOperator, deep_copy_dicts, overwrite_dicts, tensor

default_params = {
    # op related
    "op": "GATHER",
    "input_idx": None,
    "indices_idx": None,  # Add indices index
    "output_idx": None,
    # tensor related
    "input_dtype": "int32",
    "indices_dtype": "int32",  # Add indices data type
    "output_dtype": "int32",
    "input_shape": [1],  # Default shape, modify as needed
    "indices_shape": [1],  # Add indices shape
    "output_shape": [1],  # Default shape, modify as needed
}

class GatherOperator(basicOperator):
    def __init__(self, params: dict) -> None:
        super().__init__()
        self.params = deep_copy_dicts(default_params)
        overwrite_dicts(self.params, params)
        # handle input/output tensors
        self._add_input(
            self.params["input_idx"], self.params["input_dtype"], *self.params["input_shape"]
        )
        if self.params["indices_idx"] is not None:
            self._add_input(
                self.params["indices_idx"], self.params["indices_dtype"], *self.params["indices_shape"]
            )
        self._add_output(
            self.params["output_idx"], self.params["output_dtype"], *self.params["output_shape"]
        )

        if None in self.params.values():
            warnings.warn(f"parameters are not all set for op {self.params['op']}")

    def generate_inference_str(self):
        params = self.params
        input_buffer = self._getBufferstr(self.params['input1_buf_add'], self.params['input1_buf_add_offset'])
        if params["indices_idx"] is not None:
            indices_buffer = self._getBufferstr(self.params['input2_buf_add'], self.params['input2_buf_add_offset'])
        else:
            indices_buffer = "nullptr"
        output_buffer = self._getBufferstr(self.params['output_buf_add'], self.params['output_buf_add_offset'])
        input_size = params["input_shape"][0]
        num_indices = params["indices_shape"][0] if params["indices_idx"] is not None else input_size
        return f"gather({input_buffer}, {indices_buffer}, {output_buffer}, {num_indices}, {input_size});"

    def get_macs(self):
        return 0

    def get_weights_size(self):
        return 0

    def get_bias_size(self):
        return 0

    def get_scale_size(self):
        return 0

    def get_activation_size(self):
        return super().get_activation_size()

    def get_sbuf_size(self):
        return super().get_sbuf_size()

    def get_kbuf_size(self):
        return super().get_kbuf_size()
