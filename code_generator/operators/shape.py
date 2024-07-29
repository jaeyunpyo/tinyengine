import warnings
from .basic_utils import basicOperator, deep_copy_dicts, overwrite_dicts

default_params = {
    "op": "SHAPE",
    "input_idx": None,
    "output_idx": None,
    "input_dtype": "int8",
    "output_dtype": "int8",
    "input_shape": [],  # Add default empty list for input_shape
}

class ShapeOperator(basicOperator):
    ss_cnt = 0  # Class-level variable to keep track of shape value count

    def __init__(self, params: dict) -> None:
        self.params = deep_copy_dicts(default_params)
        overwrite_dicts(self.params, params)
        super().__init__()
        self._add_input(self.params["input_idx"], self.params["input_dtype"], 1)
        self._add_output(self.params["output_idx"], self.params["output_dtype"], len(self.params["input_shape"]))
        self.shape = self.params["input_shape"]

        if None in self.params.values():
            warnings.warn(f"parameters are not all set for op {self.params['op']}")

    def generate_inference_str(self):
        input_buffer = self._getBufferstr(self.params['input1_buf_add'], self.params['input1_buf_add_offset'])
        output_buffer = self._getBufferstr(self.params['output_buf_add'], self.params['output_buf_add_offset'])

        shape_str = ", ".join(map(str, self.shape))
        shape_var_name = f"shape_values{ShapeOperator.ss_cnt}"
        if self.params["input_dtype"] == "int8":
            inference_str = f"const int {shape_var_name}[] = {{{shape_str}}};\n"
        elif self.params["input_dtype"] == "int32":
            inference_str = f"const int {shape_var_name}[] = {{{shape_str}}};\n"
        inference_str += f"shape({shape_var_name}, {len(self.shape)}, {output_buffer});\n"
        
        # Increment shape value count for unique variable names
        ShapeOperator.ss_cnt += 1

        return inference_str

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
