import warnings
from .basic_utils import basicOperator, deep_copy_dicts, overwrite_dicts

__all__ = ["AvgPool2d"]

default_params = {
    # op related
    "op": "AVERAGE_POOL_2D",
    "is_SEBlock": False,
    "filter_h": None,
    "filter_w": None,
    "stride_h": None,
    "stride_w": None,
    "pad_h": None,
    "pad_w": None,
    "input_idx": None,
    "output_idx": None,
    # tensor related
    "input_dim": None,
    "input_h": None,
    "input_w": None,
    "input_c": None,
    "output_dim": None,
    "output_h": None,
    "output_w": None,
    "output_c": None,
    "kernel_h": None,
    "kernel_w": None,
    "input_dtype": "int8",
    "output_dtype": "int8",
    # trainable parameters
    "input_zero_point": None,
    "output_zero_point": None,
    "input_scale": None,
    "output_scale": None,
}


class AvgPool2d(basicOperator):
    def __init__(self, params: dict) -> None:
        self.params = deep_copy_dicts(default_params)
        overwrite_dicts(self.params, params)
        super().__init__()
        # handle input/output tensors in HWC format
        self._add_input(
            self.params["input_idx"],
            self.params["input_dtype"],
            self.params["input_c"],
            self.params["input_w"],
            self.params["input_h"],
        )
        self._add_output(
            self.params["output_idx"],
            self.params["output_dtype"],
            self.params["output_c"],
            self.params["output_w"],
            self.params["output_h"],
        )

        if None in default_params:
            warnings.warn(f"parameters are not all set for op {self.params['op']}")

    def generate_inference_str(self):
        params = self.params
        string = (
            f"avg_pooling({self._getBufferstr(params['input1_buf_add'], params['input1_buf_add_offset'])},"
            + f"{str(params['input_h'])},{str(params['input_w'])},{str(params['input_c'])},"
            + f"{str(params['filter_h'])},{str(params['filter_w'])},{str(params['stride_h'])},{str(params['stride_w'])},"
            + f"{str(params['pad_h'])},{str(params['pad_w'])},{self._getBufferstr(params['output_buf_add'], params['output_buf_add_offset'])});\n"
        )

        return string
