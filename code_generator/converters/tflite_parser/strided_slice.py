from code_generator.operators.strided_slice_v2 import StridedSliceOperator
from .utils import get_input_tensors, get_output_tensors, getTensorTypeStr
from tflite.StridedSliceOptions import StridedSliceOptions

def parse_strided_slice(op, model):
    # get_input_tensors 
    # op.InputsAsNumpy() == tensor_index_list -> _get_wrapper_tensors -> TFLiteTensorWrapper
    input_tensors = get_input_tensors(op, model)
    output_tensors = get_output_tensors(op, model)
        
    # 입력 텐서의 shape 정보 추출
    input_shapes = []
    for input_tensor in input_tensors:
        shape = input_tensor.tensor.ShapeAsNumpy().tolist()
        input_shapes.append(shape)
        # print(f"Shape: {shape}")
    
    output_shape = output_tensors[0].tensor.ShapeAsNumpy().tolist()

    # Ensure the shapes are correctly formatted
    # input_shape = input_shapes[0]
    d1 = input_shapes[0][0]
    d2 = input_shapes[1][0]
    d3 = input_shapes[2][0]
    d4 = input_shapes[3][0]
    
    if len(output_shape) < 4:
        output_shape += [1] * (4 - len(output_shape))

    # Parse options
    options = StridedSliceOptions()
    options.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)

    begin_mask = options.BeginMask()
    end_mask = options.EndMask()
    ellipsis_mask = options.EllipsisMask()
    new_axis_mask = options.NewAxisMask()
    shrink_axis_mask = options.ShrinkAxisMask()

    # Manually extract begin, end, and strides
    begin_idx = input_tensors[1].tensor.Buffer()
    end_idx = input_tensors[2].tensor.Buffer()
    strides_idx = input_tensors[3].tensor.Buffer()

    begin = model.Buffers(begin_idx).DataAsNumpy().tolist()
    end = model.Buffers(end_idx).DataAsNumpy().tolist()
    strides = model.Buffers(strides_idx).DataAsNumpy().tolist()

    params = {
        "op": "STRIDED_SLICE",
        "input_idx": input_tensors[0].tensor_idx,
        "output_idx": output_tensors[0].tensor_idx,
        "input_dtype": getTensorTypeStr(input_tensors[0].tensor.Type()),
        "output_dtype": getTensorTypeStr(output_tensors[0].tensor.Type()),
        "input_shape": input_shapes[0],
        "output_shape": output_shape,
        "d1": d1,
        "d2": d2,
        "d3": d3,
        "d4": d4,
        "o_d1": output_shape[0],
        "o_d2": output_shape[1],
        "o_d3": output_shape[2],
        "o_d4": output_shape[3],
        "begin": begin,
        "end": end,
        "strides": strides,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask
    }

    return StridedSliceOperator(params)
