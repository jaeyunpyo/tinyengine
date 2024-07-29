# code_generator/operators/reduce_prod.py

import warnings
from .basic_utils import basicOperator, deep_copy_dicts, overwrite_dicts

default_params = {
    "op": "REDUCE_PROD",
    "input_idx": None,
    "output_idx": None,
    "input_size": 1,
    "output_size": 1,
    "input_dtype": "float32",
    "output_dtype": "float32",
    "reduction_axes": [],
}

class ReduceProdOperator(basicOperator):
    _instance_counter = 0
    
    def __init__(self, params: dict) -> None:
        self.params = deep_copy_dicts(default_params)
        overwrite_dicts(self.params, params)
        super().__init__()
        self._add_input(self.params["input_idx"], self.params["input_dtype"], self.params["input_size"])
        self._add_output(self.params["output_idx"], self.params["output_dtype"], self.params["output_size"])
        
        self.instance_id = ReduceProdOperator._instance_counter
        ReduceProdOperator._instance_counter += 1
        self.reduction_axes_var_declared = False

        if None in self.params.values():
            warnings.warn(f"parameters are not all set for op {self.params['op']}")

    def generate_inference_str(self):
        params = self.params
        input_str = self._getBufferstrCast(params["input1_buf_add"], params["input1_buf_add_offset"], dtype=params["input_dtype"])
        output_str = self._getBufferstrCast(params["output_buf_add"], params["output_buf_add_offset"], dtype=params["output_dtype"])
        
        reduction_axes_var = f"reduction_axes_{self.instance_id}"
        reduction_axes_str = ', '.join(map(str, params["reduction_axes"]))

        dtype_to_ctype = {
            "float32": "float",
            "int8": "int8_t",
            "int32": "int32_t",
            # Add other types if necessary
        }

        input_ctype = dtype_to_ctype.get(params["input_dtype"], "void")
        output_ctype = dtype_to_ctype.get(params["output_dtype"], "void")

        if params["input_dtype"] == "float32":
            function_name = "reduce_prod"
        elif params["input_dtype"] == "int8":
            function_name = "reduce_prod_int8"
        elif params["input_dtype"] == "int32":
            function_name = "reduce_prod_int32"
        else:
            raise NotImplementedError(f"Data type {params['input_dtype']} is not implemented for reduce_prod operator.")

        # Declare reduction_axes array variable only once per instance
        if not self.reduction_axes_var_declared:
            declaration_str = f"int {reduction_axes_var}[] = {{{reduction_axes_str}}};\n"
            self.reduction_axes_var_declared = True           
            string = (
                f"{function_name}({input_str}, {output_str}, {params['input_size']}, "
                + f"{params['output_size']}, {reduction_axes_var}, {len(params['reduction_axes'])});\n"
            )
            return declaration_str + string
        
        else:
            string = (
                f"{function_name}({input_str}, {output_str}, {params['input_size']}, "
                + f"{params['output_size']}, {reduction_axes_var}, {len(params['reduction_axes'])});\n"
            )
            return string