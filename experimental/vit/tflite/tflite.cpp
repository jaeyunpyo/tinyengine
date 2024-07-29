#include <iostream>
#include <fstream>
#include <vector>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
// #include <tensorflow/lite/tools/optimize/calibration_reader.h>
// #include <tensorflow/lite/tools/optimize/calibration_writer.h>

std::vector<uint8_t> readInputData(const std::string& file_path, int width, int height, int channels) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        exit(1);
    }

    std::vector<uint8_t> input_data(width * height * channels);
    file.read(reinterpret_cast<char*>(input_data.data()), input_data.size());

    if (!file) {
        std::cerr << "Failed to read image data from file: " << file_path << std::endl;
        exit(1);
    }

    return input_data;
}

// void printTensorData(const TfLiteTensor* tensor) {
//     auto data = tensor->data.uint8;
//     for (int i = 0; i < tensor->bytes; ++i) {
//         std::cout << static_cast<int>(data[i]) << " ";
//     }
//     std::cout << std::endl;
// }

const char* GetOperatorName(tflite::BuiltinOperator op) {
    switch (op) {
        case tflite::BuiltinOperator_ADD: return "ADD";
        case tflite::BuiltinOperator_AVERAGE_POOL_2D: return "AVERAGE_POOL_2D";
        case tflite::BuiltinOperator_CONCATENATION: return "CONCATENATION";
        case tflite::BuiltinOperator_CONV_2D: return "CONV_2D";
        case tflite::BuiltinOperator_DEPTHWISE_CONV_2D: return "DEPTHWISE_CONV_2D";
        case tflite::BuiltinOperator_FULLY_CONNECTED: return "FULLY_CONNECTED";
        case tflite::BuiltinOperator_LOGISTIC: return "LOGISTIC";
        case tflite::BuiltinOperator_MAX_POOL_2D: return "MAX_POOL_2D";
        case tflite::BuiltinOperator_MUL: return "MUL";
        case tflite::BuiltinOperator_RELU: return "RELU";
        case tflite::BuiltinOperator_SOFTMAX: return "SOFTMAX";
        case tflite::BuiltinOperator_DEPTH_TO_SPACE: return "DEPTH_TO_SPACE";
        case tflite::BuiltinOperator_DEQUANTIZE: return "DEQUANTIZE";
        case tflite::BuiltinOperator_EMBEDDING_LOOKUP: return "EMBEDDING_LOOKUP";
        case tflite::BuiltinOperator_FLOOR: return "FLOOR";
        case tflite::BuiltinOperator_HASHTABLE_LOOKUP: return "HASHTABLE_LOOKUP";
        case tflite::BuiltinOperator_L2_NORMALIZATION: return "L2_NORMALIZATION";
        case tflite::BuiltinOperator_L2_POOL_2D: return "L2_POOL_2D";
        case tflite::BuiltinOperator_LOCAL_RESPONSE_NORMALIZATION: return "LOCAL_RESPONSE_NORMALIZATION";
        case tflite::BuiltinOperator_LSH_PROJECTION: return "LSH_PROJECTION";
        case tflite::BuiltinOperator_LSTM: return "LSTM";
        case tflite::BuiltinOperator_RELU_N1_TO_1: return "RELU_N1_TO_1";
        case tflite::BuiltinOperator_RELU6: return "RELU6";
        case tflite::BuiltinOperator_RESHAPE: return "RESHAPE";
        case tflite::BuiltinOperator_RESIZE_BILINEAR: return "RESIZE_BILINEAR";
        case tflite::BuiltinOperator_RNN: return "RNN";
        case tflite::BuiltinOperator_SPACE_TO_DEPTH: return "SPACE_TO_DEPTH";
        case tflite::BuiltinOperator_SVDF: return "SVDF";
        case tflite::BuiltinOperator_TANH: return "TANH";
        case tflite::BuiltinOperator_CONCAT_EMBEDDINGS: return "CONCAT_EMBEDDINGS";
        case tflite::BuiltinOperator_SKIP_GRAM: return "SKIP_GRAM";
        case tflite::BuiltinOperator_CALL: return "CALL";
        case tflite::BuiltinOperator_CUSTOM: return "CUSTOM";
        case tflite::BuiltinOperator_EMBEDDING_LOOKUP_SPARSE: return "EMBEDDING_LOOKUP_SPARSE";
        case tflite::BuiltinOperator_PAD: return "PAD";
        case tflite::BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN: return "UNIDIRECTIONAL_SEQUENCE_RNN";
        case tflite::BuiltinOperator_GATHER: return "GATHER";
        case tflite::BuiltinOperator_BATCH_TO_SPACE_ND: return "BATCH_TO_SPACE_ND";
        case tflite::BuiltinOperator_SPACE_TO_BATCH_ND: return "SPACE_TO_BATCH_ND";
        case tflite::BuiltinOperator_TRANSPOSE: return "TRANSPOSE";
        case tflite::BuiltinOperator_MEAN: return "MEAN";
        case tflite::BuiltinOperator_SUB: return "SUB";
        case tflite::BuiltinOperator_DIV: return "DIV";
        case tflite::BuiltinOperator_SQUEEZE: return "SQUEEZE";
        case tflite::BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM: return "UNIDIRECTIONAL_SEQUENCE_LSTM";
        case tflite::BuiltinOperator_STRIDED_SLICE: return "STRIDED_SLICE";
        case tflite::BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN: return "BIDIRECTIONAL_SEQUENCE_RNN";
        case tflite::BuiltinOperator_EXP: return "EXP";
        case tflite::BuiltinOperator_TOPK_V2: return "TOPK_V2";
        case tflite::BuiltinOperator_SPLIT: return "SPLIT";
        case tflite::BuiltinOperator_LOG_SOFTMAX: return "LOG_SOFTMAX";
        case tflite::BuiltinOperator_DELEGATE: return "DELEGATE";
        case tflite::BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM: return "BIDIRECTIONAL_SEQUENCE_LSTM";
        case tflite::BuiltinOperator_CAST: return "CAST";
        case tflite::BuiltinOperator_PRELU: return "PRELU";
        case tflite::BuiltinOperator_MAXIMUM: return "MAXIMUM";
        case tflite::BuiltinOperator_ARG_MAX: return "ARG_MAX";
        case tflite::BuiltinOperator_MINIMUM: return "MINIMUM";
        case tflite::BuiltinOperator_LESS: return "LESS";
        case tflite::BuiltinOperator_NEG: return "NEG";
        case tflite::BuiltinOperator_PADV2: return "PADV2";
        case tflite::BuiltinOperator_GREATER: return "GREATER";
        case tflite::BuiltinOperator_GREATER_EQUAL: return "GREATER_EQUAL";
        case tflite::BuiltinOperator_LESS_EQUAL: return "LESS_EQUAL";
        case tflite::BuiltinOperator_SELECT: return "SELECT";
        case tflite::BuiltinOperator_SLICE: return "SLICE";
        case tflite::BuiltinOperator_SIN: return "SIN";
        case tflite::BuiltinOperator_TRANSPOSE_CONV: return "TRANSPOSE_CONV";
        case tflite::BuiltinOperator_SPARSE_TO_DENSE: return "SPARSE_TO_DENSE";
        case tflite::BuiltinOperator_TILE: return "TILE";
        case tflite::BuiltinOperator_EXPAND_DIMS: return "EXPAND_DIMS";
        case tflite::BuiltinOperator_EQUAL: return "EQUAL";
        case tflite::BuiltinOperator_NOT_EQUAL: return "NOT_EQUAL";
        case tflite::BuiltinOperator_LOG: return "LOG";
        case tflite::BuiltinOperator_SUM: return "SUM";
        case tflite::BuiltinOperator_SQRT: return "SQRT";
        case tflite::BuiltinOperator_RSQRT: return "RSQRT";
        case tflite::BuiltinOperator_SHAPE: return "SHAPE";
        case tflite::BuiltinOperator_POW: return "POW";
        case tflite::BuiltinOperator_ARG_MIN: return "ARG_MIN";
        case tflite::BuiltinOperator_FAKE_QUANT: return "FAKE_QUANT";
        case tflite::BuiltinOperator_REDUCE_PROD: return "REDUCE_PROD";
        case tflite::BuiltinOperator_REDUCE_MAX: return "REDUCE_MAX";
        case tflite::BuiltinOperator_PACK: return "PACK";
        case tflite::BuiltinOperator_LOGICAL_OR: return "LOGICAL_OR";
        case tflite::BuiltinOperator_ONE_HOT: return "ONE_HOT";
        case tflite::BuiltinOperator_LOGICAL_AND: return "LOGICAL_AND";
        case tflite::BuiltinOperator_LOGICAL_NOT: return "LOGICAL_NOT";
        case tflite::BuiltinOperator_UNPACK: return "UNPACK";
        case tflite::BuiltinOperator_REDUCE_MIN: return "REDUCE_MIN";
        case tflite::BuiltinOperator_FLOOR_DIV: return "FLOOR_DIV";
        case tflite::BuiltinOperator_REDUCE_ANY: return "REDUCE_ANY";
        case tflite::BuiltinOperator_SQUARE: return "SQUARE";
        case tflite::BuiltinOperator_ZEROS_LIKE: return "ZEROS_LIKE";
        case tflite::BuiltinOperator_FILL: return "FILL";
        case tflite::BuiltinOperator_FLOOR_MOD: return "FLOOR_MOD";
        case tflite::BuiltinOperator_RANGE: return "RANGE";
        case tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR: return "RESIZE_NEAREST_NEIGHBOR";
        case tflite::BuiltinOperator_LEAKY_RELU: return "LEAKY_RELU";
        case tflite::BuiltinOperator_SQUARED_DIFFERENCE: return "SQUARED_DIFFERENCE";
        case tflite::BuiltinOperator_MIRROR_PAD: return "MIRROR_PAD";
        case tflite::BuiltinOperator_ABS: return "ABS";
        case tflite::BuiltinOperator_SPLIT_V: return "SPLIT_V";
        case tflite::BuiltinOperator_UNIQUE: return "UNIQUE";
        case tflite::BuiltinOperator_CEIL: return "CEIL";
        case tflite::BuiltinOperator_REVERSE_V2: return "REVERSE_V2";
        case tflite::BuiltinOperator_ADD_N: return "ADD_N";
        case tflite::BuiltinOperator_GATHER_ND: return "GATHER_ND";
        case tflite::BuiltinOperator_COS: return "COS";
        case tflite::BuiltinOperator_WHERE: return "WHERE";
        case tflite::BuiltinOperator_RANK: return "RANK";
        case tflite::BuiltinOperator_ELU: return "ELU";
        case tflite::BuiltinOperator_REVERSE_SEQUENCE: return "REVERSE_SEQUENCE";
        case tflite::BuiltinOperator_MATRIX_DIAG: return "MATRIX_DIAG";
        case tflite::BuiltinOperator_QUANTIZE: return "QUANTIZE";
        case tflite::BuiltinOperator_MATRIX_SET_DIAG: return "MATRIX_SET_DIAG";
        case tflite::BuiltinOperator_ROUND: return "ROUND";
        case tflite::BuiltinOperator_HARD_SWISH: return "HARD_SWISH";
        case tflite::BuiltinOperator_IF: return "IF";
        case tflite::BuiltinOperator_WHILE: return "WHILE";
        case tflite::BuiltinOperator_NON_MAX_SUPPRESSION_V4: return "NON_MAX_SUPPRESSION_V4";
        case tflite::BuiltinOperator_NON_MAX_SUPPRESSION_V5: return "NON_MAX_SUPPRESSION_V5";
        case tflite::BuiltinOperator_SCATTER_ND: return "SCATTER_ND";
        case tflite::BuiltinOperator_SELECT_V2: return "SELECT_V2";
        case tflite::BuiltinOperator_DENSIFY: return "DENSIFY";
        case tflite::BuiltinOperator_SEGMENT_SUM: return "SEGMENT_SUM";
        case tflite::BuiltinOperator_BATCH_MATMUL: return "BATCH_MATMUL";
        case tflite::BuiltinOperator_CUMSUM: return "CUMSUM";
        case tflite::BuiltinOperator_CALL_ONCE: return "CALL_ONCE";
        case tflite::BuiltinOperator_BROADCAST_TO: return "BROADCAST_TO";
        case tflite::BuiltinOperator_RFFT2D: return "RFFT2D";
        case tflite::BuiltinOperator_CONV_3D: return "CONV_3D";
        case tflite::BuiltinOperator_IMAG: return "IMAG";
        case tflite::BuiltinOperator_REAL: return "REAL";
        case tflite::BuiltinOperator_COMPLEX_ABS: return "COMPLEX_ABS";
        case tflite::BuiltinOperator_HASHTABLE: return "HASHTABLE";
        case tflite::BuiltinOperator_HASHTABLE_FIND: return "HASHTABLE_FIND";
        case tflite::BuiltinOperator_HASHTABLE_IMPORT: return "HASHTABLE_IMPORT";
        case tflite::BuiltinOperator_HASHTABLE_SIZE: return "HASHTABLE_SIZE";
        case tflite::BuiltinOperator_REDUCE_ALL: return "REDUCE_ALL";
        case tflite::BuiltinOperator_CONV_3D_TRANSPOSE: return "CONV_3D_TRANSPOSE";
        case tflite::BuiltinOperator_VAR_HANDLE: return "VAR_HANDLE";
        case tflite::BuiltinOperator_READ_VARIABLE: return "READ_VARIABLE";
        case tflite::BuiltinOperator_ASSIGN_VARIABLE: return "ASSIGN_VARIABLE";
        default: return "UNKNOWN";
    }
}

void printTensorParams(const TfLiteTensor* tensor) {
    std::cout << "Type: " << tensor->type << std::endl;
    std::cout << "Shape: ";
    for (int i = 0; i < tensor->dims->size; ++i) {
        std::cout << tensor->dims->data[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Bytes: " << tensor->bytes << std::endl;
    std::cout << "Data: ";
    auto data = tensor->data.uint8;
    for (int i = 0; i < tensor->bytes; ++i) {
        std::cout << static_cast<int>(data[i]) << " ";
    }
    std::cout << std::endl;
    // QNN parameters
    if (tensor->quantization.type != kTfLiteNoQuantization) {
        const auto& quant_params = tensor->params;
        std::cout << "Quantization - Scale: " << quant_params.scale << ", Zero Point: " << quant_params.zero_point << std::endl;
    }
    std::cout << std::endl;
}

// void printOperatorInfo(tflite::Interpreter* interpreter) {
//     const auto& execution_plan = interpreter->execution_plan();
//     for (size_t i = 0; i < execution_plan.size(); ++i) {
//         int node_index = execution_plan[i];
//         const auto* node_and_reg = interpreter->node_and_registration(node_index);
//         const TfLiteNode* node = &node_and_reg->first;
//         const TfLiteRegistration* reg = &node_and_reg->second;

//         std::cout << "Layer " << i << ":\n";
//         std::cout << "Node index: " << node_index << "\n";
//         std::cout << "Opcode: " << reg->builtin_code << " (" << tflite::EnumNameBuiltinOperator(static_cast<tflite::BuiltinOperator>(reg->builtin_code)) << ")\n";

//         for (int j = 0; j < node->inputs->size; ++j) {
//             int tensor_index = node->inputs->data[j];
//             std::cout << "Input tensor " << j << ":\n";
//             printTensorParams(interpreter->tensor(tensor_index));
//         }

//         for (int j = 0; j < node->outputs->size; ++j) {
//             int tensor_index = node->outputs->data[j];
//             std::cout << "Output tensor " << j << ":\n";
//             printTensorParams(interpreter->tensor(tensor_index));
//         }
//     }
// }

int main() {
    const std::string model_path = "deit_tiny_patch16_224_int8.tflite";
    const std::string input_image_path = "test_image.bin";
    const int image_width = 224;
    const int image_height = 224;
    const int image_channels = 3;

    std::vector<uint8_t> input_data = readInputData(input_image_path, image_width, image_height, image_channels);

    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model) {
        std::cerr << "Failed to load model: " << model_path << std::endl;
        return 1;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        std::cerr << "Failed to create interpreter." << std::endl;
        return 1;
    }

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors." << std::endl;
        return 1;
    }

    // Disable graph optimizations by setting the interpreter options
    interpreter->SetNumThreads(1); // Use single-threaded execution
    // interpreter->UseNNAPI(false);  // Disable NNAPI delegate

    int input_tensor_index = interpreter->inputs()[0];
    TfLiteTensor* input_tensor = interpreter->tensor(input_tensor_index);

    std::memcpy(input_tensor->data.uint8, input_data.data(), input_data.size() * sizeof(uint8_t));
    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Failed to invoke TFLite interpreter." << std::endl;
        return -1;
    }

    int model_size = interpreter->execution_plan().size();
    std::cout << "Model size: " << model_size << std::endl;

    for (size_t i = 0; i < model_size ; ++i) {       

        int node_index = interpreter->execution_plan()[i];
        const auto* node_and_reg = interpreter->node_and_registration(node_index);
        const TfLiteNode* node = &node_and_reg->first;
        const TfLiteRegistration* reg = &node_and_reg->second;

        std::cout << "Layer " << i << ":\n";
        std::cout << "Node index: " << node_index << "\n";  // 추가된 디버깅 출력
        std::cout << "Opcode: " << reg->builtin_code << " (" << GetOperatorName(static_cast<tflite::BuiltinOperator>(reg->builtin_code)) << ")\n";

        for (int j = 0; j < node->inputs->size; ++j) {
            const TfLiteTensor* input_tensor = interpreter->tensor(node->inputs->data[j]);
            std::cout << "Input tensor " << j << ":\n";
            printTensorParams(input_tensor);
            // printTensorData(input_tensor);
        }
        for (int j = 0; j < node->outputs->size; ++j) {
            const TfLiteTensor* output_tensor = interpreter->tensor(node->outputs->data[j]);
            std::cout << "Output tensor " << j << ":\n";
            printTensorParams(output_tensor);
            // printTensorData(output_tensor);
        }
        std::cout << std::endl;
    }

    return 0;
}
