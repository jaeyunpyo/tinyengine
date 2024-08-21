#include <cstring>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric> // For std::iota

typedef int8_t q7_t;
typedef uint8_t q8_t;
typedef int16_t q15_t;
typedef uint16_t q16_t;
typedef int32_t q31_t;
typedef uint32_t q32_t;

typedef struct add_params {
    int input_h, input_w, input_c, left_shift;
    int input1_offset, input1_multiplier, input1_shift;
    int input2_offset, input2_multiplier, input2_shift;
    int output_offset, output_multiplier, output_shift;
    int quantized_activation_max, quantized_activation_min;

} ADD_params;
using namespace std;

// 양자화된 텐서의 덧셈을 수행하는 함수
// tinyengine
void add_fpreq(int size, const int8_t* input1_data, const float input1_scale, const float input1_zero,
                            const int8_t* input2_data, const float input2_scale, const float input2_zero,
                            const float output_scale, const float zero_y, int8_t* output_data) {
    for (int i = 0; i < size; ++i) {
        float input1_fp = ((float)*input1_data++ - input1_zero) * input1_scale;
        float input2_fp = ((float)*input2_data++ - input2_zero) * input2_scale;
        int clamped_output =
            (int)round((input1_fp + input2_fp) / output_scale + zero_y);  // to align with tvm implementation
        clamped_output = TN_MAX(clamped_output, -128);
        clamped_output = TN_MIN(clamped_output, 127);
        output_data[i] = (int8_t)(clamped_output);
    }

    //return STATE_SUCCESS;
}

// avg_pooling 함수의 정의
// tinyen
// void avg_pooling(uint8_t* input, int input_h, int input_w, int input_c,
//                  int filter_h, int filter_w, int stride_h, int stride_w,
//                  int pad_h, int pad_w,
//                  uint8_t* output) {
// avg_pooling(&buffer0[150528], 192,1,197, 
//             192,1,1,1,
//             0,0,
//             &buffer0[37824]);
void avg_pooling(const int8_t* input, const uint16_t input_h, const uint16_t input_w, const uint16_t input_c,
                              const uint16_t sample_h, const uint16_t sample_w, const uint16_t output_h,
                              const uint16_t output_w, const int32_t out_activation_min,
                              const int32_t out_activation_max, int8_t* output) {
    int h, w, c;
    int sh, sw;
    const int divider_half = ((sample_h * sample_w) / 2);
    for (c = 0; c < input_c; c++) {
        for (h = 0; h < output_h; h++) {
            for (w = 0; w < output_w; w++) {
                int avg = 0;

                for (sh = 0; sh < sample_h; sh++) {
                    int height = sh + h * sample_h;
                    for (sw = 0; sw < sample_w; sw++) {
                        int width = sw + w * sample_w;
                        avg += input[(width + height * input_w) * input_c + c];
                    }
                }

                // for rounded div
                if (avg > 0)
                    avg += divider_half;
                else
                    avg -= divider_half;

                int out = avg / (sample_h * sample_w);
                out = TN_MAX(out, out_activation_min);
                out = TN_MIN(out, out_activation_max);
                output[(w + h * output_w) * input_c + c] = out;
            }
        }
    }
    //return STATE_SUCCESS;
}

// batch_matmul 함수의 정의
void batch_matmul(int8_t* input, int8_t* input2, int8_t* output, 
                  int batch_size, int M, int K, int N, 
                  bool adj_x, bool adj_y,
                  int input_zero_point, int input2_zero_point, int output_zero_point,
                  float input_scale, float input2_scale, float output_scale,
                  int input_shift, int input2_shift, int output_shift,
                  int input_multiplier, int input2_multiplier, int output_multiplier) {
    for (int batch = 0; batch < batch_size; ++batch) {
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                int32_t acc = 0;
                for (int k = 0; k < K; ++k) {
                    int a_index = adj_x ? (batch * M * K + k * M + m) : (batch * M * K + m * K + k);
                    int b_index = adj_y ? (batch * K * N + n * K + k) : (batch * K * N + k * N + n);
                    
                    int32_t a_val = input[a_index] - input_zero_point;
                    int32_t b_val = input2[b_index] - input2_zero_point;

                    // Quantize multiplication
                    a_val = ((a_val * input_multiplier) + (1 << (input_shift - 1))) >> input_shift;
                    b_val = ((b_val * input2_multiplier) + (1 << (input2_shift - 1))) >> input2_shift;

                    acc += a_val * b_val;
                }
                
                // Quantize the accumulator
                acc = ((acc * output_multiplier) + (1 << (output_shift - 1))) >> output_shift;
                acc += output_zero_point;
                acc = std::max(std::min(acc, 127), -128);  // Clamp to int8 range

                output[batch * M * N + m * N + n] = static_cast<int8_t>(acc);
            }
        }
    }
}


// concatenate 함수의 정의
void concatenate(const int8_t* input1, const int8_t* input2, int size1, int size2, int8_t* output, int axis) {
    if (axis != 0) {
        cerr << "Only axis = 0 is supported in this simple implementation." << endl;
        return;
    }

    std::memcpy(output, input1, sizeof(int32_t) * size1);
    std::memcpy(output + size1, input2, sizeof(int32_t) * size2);
}
void concatenate(const int32_t* input1, const int32_t* input2, int size1, int size2, int32_t* output, int axis) {
    if (axis != 0) {
        cerr << "Only axis = 0 is supported in this simple implementation." << endl;
        return;
    }

    std::memcpy(output, input1, sizeof(int32_t) * size1);
    std::memcpy(output + size1, input2, sizeof(int32_t) * size2);
}
void concatenate(const uint8_t* input1, const uint8_t* input2, int size1, int size2, uint8_t* output, int axis) {
    if (axis != 0) {
        cerr << "Only axis = 0 is supported in this simple implementation." << endl;
        return;
    }

    std::memcpy(output, input1, sizeof(uint8_t) * size1);
    std::memcpy(output + size1, input2, sizeof(uint8_t) * size2);
}

// printf("/* layer 5:CONV_2D */\n");
// conv2d_16x16_fpreq(
//     &buffer0[0],224,224,3,
//     (const q7_t*) 0,0,scales0,
//     14,14,192,
//     &buffer0[0],16,8,
//     127,-68,-128,127,
//     sbuf);
// conv2d_16x16_fpreq 함수의 정의
#include <immintrin.h>  // For AVX2 instructions
void conv2d_16x16_fpreq(const int8_t* input, int input_w, int input_h, int input_c,
                        const int8_t* weights, const int32_t* bias, const float* scales,
                        int output_w, int output_h, int output_c,
                        int8_t* output, int stride, int pad, 
                        int input_zero_point, int output_zero_point, int min_val, int max_val, 
                        int16_t* sbuf)
{
    // 패딩을 적용한 입력 크기를 계산합니다.
    int padded_input_w = input_w + 2 * pad;
    int padded_input_h = input_h + 2 * pad;

    // 임시 버퍼를 초기화합니다.
    std::fill(sbuf, sbuf + padded_input_w * padded_input_h * input_c, input_zero_point);

    // 패딩을 적용하여 입력을 임시 버퍼에 복사합니다.
    // (1,3,224,224) -> (1,224,224,3)
    for (int c = 0; c < input_c; ++c) {
        for (int h = 0; h < input_h; ++h) {
            for (int w = 0; w < input_w; ++w) {
                sbuf[(c * padded_input_h + (h + pad)) * padded_input_w + (w + pad)] = input[(h * input_w + w) * input_c + c];
            }
        }
    }

    // 16x16 패치 Conv2D 연산을 수행합니다.
    int kernel_size = 16;

    for (int oc = 0; oc < output_c; ++oc) {
        for (int oh = 0; oh < output_h; ++oh) {
            for (int ow = 0; ow < output_w; ++ow) {
                __m256i acc = _mm256_set1_epi32(bias[oc]);

                for (int ic = 0; ic < input_c; ++ic) {
                    for (int kh = 0; kh < kernel_size; kh += 4) {  // Loop unrolling
                        for (int kw = 0; kw < kernel_size; kw += 4) {
                            int h_in = oh * stride + kh;
                            int w_in = ow * stride + kw;

                            // Load input values
                            __m256i input_values = _mm256_set_epi32(
                                sbuf[(ic * padded_input_h + h_in + 3) * padded_input_w + w_in + 3],
                                sbuf[(ic * padded_input_h + h_in + 2) * padded_input_w + w_in + 2],
                                sbuf[(ic * padded_input_h + h_in + 1) * padded_input_w + w_in + 1],
                                sbuf[(ic * padded_input_h + h_in) * padded_input_w + w_in],
                                sbuf[(ic * padded_input_h + h_in + 3) * padded_input_w + w_in + 3],
                                sbuf[(ic * padded_input_h + h_in + 2) * padded_input_w + w_in + 2],
                                sbuf[(ic * padded_input_h + h_in + 1) * padded_input_w + w_in + 1],
                                sbuf[(ic * padded_input_h + h_in) * padded_input_w + w_in]
                            );

                            // Load weight values
                            __m256i weight_values = _mm256_set_epi32(
                                weights[(oc * input_c + ic) * kernel_size * kernel_size + (kh + 3) * kernel_size + kw + 3],
                                weights[(oc * input_c + ic) * kernel_size * kernel_size + (kh + 2) * kernel_size + kw + 2],
                                weights[(oc * input_c + ic) * kernel_size * kernel_size + (kh + 1) * kernel_size + kw + 1],
                                weights[(oc * input_c + ic) * kernel_size * kernel_size + kh * kernel_size + kw],
                                weights[(oc * input_c + ic) * kernel_size * kernel_size + (kh + 3) * kernel_size + kw + 3],
                                weights[(oc * input_c + ic) * kernel_size * kernel_size + (kh + 2) * kernel_size + kw + 2],
                                weights[(oc * input_c + ic) * kernel_size * kernel_size + (kh + 1) * kernel_size + kw + 1],
                                weights[(oc * input_c + ic) * kernel_size * kernel_size + kh * kernel_size + kw]
                            );

                            // Perform the multiply-add operation
                            __m256i mul = _mm256_mullo_epi32(input_values, weight_values);
                            acc = _mm256_add_epi32(acc, mul);
                        }
                    }
                }

                // Sum the accumulator vector
                int32_t acc_array[8];
                _mm256_storeu_si256((__m256i*)acc_array, acc);
                int32_t sum = 0;
                for (int i = 0; i < 8; ++i) {
                    sum += acc_array[i];
                }

                // Apply the scale and zero points, then clamp the result
                sum = static_cast<int32_t>(sum * scales[oc]);
                sum += output_zero_point;
                sum = std::max(min_val, std::min(max_val, sum));
                output[(oc * output_h + oh) * output_w + ow] = static_cast<int8_t>(sum);
            }
        }
    }

    // 출력 형상을 출력합니다.
    std::cout << "Output shape: (1, " << output_h << ", " << output_w << ", " << output_c << ")\n";

}


//fully_connected(
// (signed char*)&buffer0[264000], (signed char*)&buffer0[4975536], NULL, (signed char*)&buffer0[4975536],
//  -73, 0, 14, 
//  0.05217108875513077, 0.0033674186561256647, 0.036277152597904205, 
//  0, 0, 0, 
//  0, 0, 0, 
//  0);

// fully_connected(
// (signed char*)&buffer0[264000], (signed char*)&buffer0[4975536], NULL, (signed char*)&buffer0[4975536],
//  -73, 0, 14, 
//  0.05217108875513077, 0.0033674186561256647, 0.036277152597904205, 
//  0, 0, 0, 
//  0, 0, 0, 
//  0, 1, 192);


// fully_connected 함수의 정의
void fully_connected(const int8_t* input, const int8_t* weights, const int8_t* bias, int8_t* output,
                     int input_zero_point, int weight_zero_point, int output_zero_point,
                     float input_scale, float weight_scale, float output_scale,
                     int input_multiplier, int weight_multiplier, int output_multiplier,
                     int input_shift, int weight_shift, int output_shift,
                     int activation, int input_size, int output_size) {
    for (int out_idx = 0; out_idx < output_size; ++out_idx) {
        int32_t acc = 0;

        for (int in_idx = 0; in_idx < input_size; ++in_idx) {
            int32_t input_val = input[in_idx] - input_zero_point;
            int32_t weight_val = weights[out_idx * input_size + in_idx] - weight_zero_point;
            acc += input_val * weight_val;
        }

        if (bias) {
            acc += bias[out_idx];
        }

        float real_multiplier = input_scale * weight_scale / output_scale;
        acc = static_cast<int32_t>(std::round(acc * real_multiplier));

        acc = acc + output_zero_point;

        if (activation == 0) {
            // No activation
        } else if (activation == 1) {
            // ReLU activation
            acc = std::max(0, acc);
        } else if (activation == 2) {
            // ReLU6 activation
            acc = std::max(0, std::min(6, acc));
        }

        acc = std::min(std::max(acc, -127), 127);
        output[out_idx] = static_cast<int8_t>(acc);
    }
}

// gather 함수의 정의
void gather(const int8_t* input, const int8_t* indices, int8_t* output, int num_indices, int input_size) {
    for (int i = 0; i < num_indices; ++i) {
        int index = static_cast<int>(indices[i]);
        if (index < 0 || index >= input_size) {
            // Handle out-of-bound indices
            continue;
        }
        output[i] = input[index];
    }
}

// mul_int8 함수의 정의
void mul_int8(int size, const int8_t* input1, const int8_t* input2, int8_t* output,
              int input1_zero_point, int input2_zero_point, int output_zero_point,
              float input1_scale, float input2_scale, float output_scale,
              int input1_shift, int input2_shift, int output_shift) {
    for (int i = 0; i < size; ++i) {
        // 입력 값을 양자화된 값으로 변환
        int32_t input1_val = (input1[i] - input1_zero_point);
        int32_t input2_val = (input2[i] - input2_zero_point);

        // 입력 값에 스케일 적용
        int32_t scaled_input1 = input1_val * (1 << input1_shift);
        int32_t scaled_input2 = input2_val * (1 << input2_shift);

        // 두 입력 값을 곱함
        int32_t real_output = scaled_input1 * scaled_input2;

        // 결과 값을 스케일 및 시프트 적용하여 양자화
        int32_t quantized_output = static_cast<int32_t>((real_output * output_scale) / (1 << output_shift));
        quantized_output += output_zero_point;

        // 값 제한 (클램핑)
        quantized_output = std::min(127, std::max(-128, quantized_output));
        output[i] = static_cast<int8_t>(quantized_output);
    }
}

// 두 입력만을 처리하는 pack 함수의 정의
template <typename T>
void pack(const T* input1, const T* input2, T* output, int num_inputs, int axis) {
    const T* inputs[] = {input1, input2};
    int input_size = 1;  // Assuming all inputs have the same size
    int output_stride = input_size * num_inputs;

    int output_offset = 0;
    for (int i = 0; i < num_inputs; ++i) {
        const T* input = inputs[i];
        for (int j = 0; j < input_size; ++j) {
            int out_idx = output_offset + j * num_inputs + i;
            output[out_idx] = input[j];
        }
        output_offset += input_size;
    }
}

// pack 함수의 정의
template <typename T>
void pack(const T* input1, const T* input2, const T* input3, T* output, int num_inputs, int axis) {
    const T* inputs[] = {input1, input2, input3};
    int input_size = 1;  // Assuming all inputs have the same size
    int output_stride = input_size * num_inputs;

    int output_offset = 0;
    //printf("num_inputs: %d\n", num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
        //printf("i: %d\n", i);
        const T* input = inputs[i];
        //printf("input: %d\n", input[0]);
        for (int j = 0; j < input_size; ++j) {
            //printf("\tj: %d\n", j);
            int out_idx = output_offset + j * num_inputs + i;
            //printf("\tout_idx: %d\n", out_idx);
            output[out_idx] = input[j];            
        }
        output_offset += input_size;
    }
}

template <typename T>
void pack(const T* input1, const T* input2, const T* input3, const T* input4, T* output, int num_inputs, int axis) {
    const T* inputs[] = {input1, input2, input3, input4};
    int input_size = 1;  // Assuming all inputs have the same size
    int output_stride = input_size * num_inputs;

    int output_offset = 0;
    //printf("num_inputs: %d\n", num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
        //printf("i: %d\n", i);
        const T* input = inputs[i];
        //printf("input: %d\n", input[0]);
        for (int j = 0; j < input_size; ++j) {
            //printf("\tj: %d\n", j);
            int out_idx = output_offset + j * num_inputs + i;
            //printf("\tout_idx: %d\n", out_idx);
            output[out_idx] = input[j];            
        }
        output_offset += input_size;
    }
}

// placeholder_for_greater_op_codes 함수의 정의
void placeholder_for_greater_op_codes(const uint8_t* input, uint8_t* output, int input_dim1, int input_dim2) {
    // 입력 텐서를 출력 텐서로 단순히 복사하는 기본 구현
    for (int i = 0; i < input_dim1; ++i) {
        for (int j = 0; j < input_dim2; ++j) {
            output[i * input_dim2 + j] = input[i * input_dim2 + j];
        }
    }
}

std::vector<int> validate_reduction_axes(const int* reduction_axes, int num_reduction_axes, int default_value) {
    std::vector<int> valid_axes;
    for (int i = 0; i < num_reduction_axes; ++i) {
        if (reduction_axes[i] == 0) {
            valid_axes.push_back(default_value);
        } else {
            valid_axes.push_back(reduction_axes[i]);
        }
    }
    return valid_axes;
}
// reduce_prod_int32 함수의 정의
void reduce_prod_int32(const int32_t* input, int32_t* output, 
                       int input_size, int output_size, 
                       const int* reduction_axes, int num_reduction_axes) {
    // reduction_axes 유효성 검사 및 기본 값 설정
    std::vector<int> valid_reduction_axes = validate_reduction_axes(reduction_axes, num_reduction_axes, 1);

    // 초기화: 출력 배열의 모든 요소를 1로 설정
    for (int i = 0; i < output_size; ++i) {
        output[i] = 1;
    }

    // 각 입력 요소를 순회하며 곱셈을 수행
    for (int i = 0; i < input_size; ++i) {
        int output_index = 0;
        bool is_zero = false;

        // 입력 인덱스를 출력 인덱스로 매핑
        for (int j = 0; j < num_reduction_axes; ++j) {
            output_index += (i / valid_reduction_axes[j]) % output_size;
            if (input[i] == 0) {
                is_zero = true;
                break;
            }
        }

        // 입력 값이 0인 경우 처리
        if (is_zero) {
            output[output_index] = 0;
        } else {
            // 입력 값을 출력 배열의 해당 요소에 곱함
            output[output_index] *= input[i];
        }
    }
}

// // rsqrt 함수의 정의 
// void rsqrt(const int8_t* input, int8_t* output, int rows, int cols) {
//     for (int i = 0; i < rows; ++i) {
//         for (int j = 0; j < cols; ++j) {
//             int index = i * cols + j;
//             output[index] = 1.0f / std::sqrt(input[index]);
//         }
//     }
// }

// Approximation of 1/sqrt(x) using linear interpolation
float rsqrt_approx(float number) {
    const float threehalfs = 1.5F;
    float x2 = number * 0.5F;
    float y = number;

    // Evil floating point bit level hacking
    long i = *(long*)&y;
    i = 0x5f3759df - (i >> 1);
    y = *(float*)&i;

    // 1st iteration of Newton's method
    y = y * (threehalfs - (x2 * y * y));
    return y;
}

void rsqrt(const int8_t* input, int8_t* output, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int index = i * cols + j;
            float value = (float)input[index];
            float approx = rsqrt_approx(value);
            output[index] = (int8_t)(approx * 128); // Scale for int8_t range
        }
    }
}

// shape 함수의 정의
void shape(const int* input_tensor, int num_dimensions, int* output_tensor) {
    // input_tensor: 입력 텐서의 shape 정보를 나타내는 배열입니다.
    // num_dimensions: 입력 텐서의 차원 수입니다.
    // output_tensor: 출력 텐서로, shape 정보를 저장할 배열입니다.

    // 입력 텐서의 shape 정보를 출력 텐서에 복사합니다.
    for (int i = 0; i < num_dimensions; ++i) {
        output_tensor[i] = input_tensor[i];
    }
}


// 고정 소수점 기반 소프트맥스 함수
void softmax(const int8_t* input, int8_t* output, int batch_size, int num_classes, int beta) {
    const int32_t kScalingFactor = 1 << 20; // 고정 소수점 비트 수 조정
    const int32_t kMaxExponent = 10; // 지수 함수 최대값 제한
    const int32_t kMinExponent = -20; // 지수 함수 최소값 제한

    for (int b = 0; b < batch_size; ++b) {
        const int8_t* input_batch = input + b * num_classes;
        int8_t* output_batch = output + b * num_classes;

        // 입력 텐서의 최대값 찾기
        int8_t max_val = *std::max_element(input_batch, input_batch + num_classes);

        // 지수 함수의 합 계산
        int32_t sum = 0;
        int32_t* exp_values = new int32_t[num_classes];
        for (int i = 0; i < num_classes; ++i) {
            int32_t scaled_input = (input_batch[i] - max_val) * beta;
            scaled_input = std::max(kMinExponent, std::min(kMaxExponent, scaled_input)); // 범위 제한

            exp_values[i] = std::exp(scaled_input / static_cast<float>(kScalingFactor));
            sum += exp_values[i];
        }

        // 출력 텐서에 소프트맥스 값 저장
        for (int i = 0; i < num_classes; ++i) {
            output_batch[i] = static_cast<int8_t>((exp_values[i] * 128 / sum) - 128); // 고정 소수점으로 변환
        }

        delete[] exp_values;
    }
}
// squared_difference(
// (signed char*)&buffer0[150528], 
// (signed char*)&buffer0[188352], 
// (signed char*)&buffer0[188352], 
// 1, 
// 197)
// squared_difference 함수의 정의
void squared_difference(const int8_t* input1, const int8_t* input2, int8_t* output, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int index = i * cols + j;
            signed char diff = input1[index] - input2[index];
            output[index] = diff * diff;
        }
    }
}

// float32 타입을 위한 strided slice 함수
void strided_slice_4Dto4D(const float* input, int d1, int d2, int d3, int d4,
                              const int* begin, const int* end, const int* strides,
                              float* output, int o_d1, int o_d2, int o_d3, int o_d4,
                              int begin_mask, int end_mask, int ellipsis_mask,
                              int new_axis_mask, int shrink_axis_mask) {
    // 구현 로직
    // 여기서는 간단히 입력 텐서의 일부를 출력 텐서로 복사하는 예제를 제공합니다.
    int input_size = d1 * d2 * d3 * d4;
    int output_size = o_d1 * o_d2 * o_d3 * o_d4;

    for (int i = 0; i < output_size; ++i) {
        int idx = begin[0] + i * strides[0];
        if (idx < end[0]) {
            output[i] = input[idx];
        } else {
            break;
        }
    }
}

// int8 타입을 위한 strided slice 함수
void strided_slice_4Dto4D_int8(const int8_t* input, int d1, int d2, int d3, int d4,
                               const int8_t* begin, const int8_t* end, const int8_t* strides,
                               int8_t* output, int o_d1, int o_d2, int o_d3, int o_d4,
                               int begin_mask, int end_mask, int ellipsis_mask,
                               int new_axis_mask, int shrink_axis_mask) {
    // Helper function to apply masks
    auto apply_mask = [begin_mask, end_mask](int idx, int dim, int mask, int bit) {
        return (mask & (1 << bit)) ? (mask == begin_mask ? 0 : dim) : idx;
    };

    // Initialize the start, stop, and stride arrays
    int start[4] = {0};
    int stop[4] = {0};
    int stride[4] = {0};
    int dims[4] = {d1, d2, d3, d4};

    for (int i = 0; i < 4; ++i) {
        start[i] = apply_mask(begin[i], dims[i], begin_mask, i);
        stop[i] = apply_mask(end[i], dims[i], end_mask, i);
        stride[i] = strides[i];
    }

    // Perform the slice
    int out_index = 0;
    for (int i = start[0]; i < stop[0]; i += stride[0]) {
        for (int j = start[1]; j < stop[1]; j += stride[1]) {
            for (int k = start[2]; k < stop[2]; k += stride[2]) {
                for (int l = start[3]; l < stop[3]; l += stride[3]) {
                    int in_index = i * d2 * d3 * d4 + j * d3 * d4 + k * d4 + l;
                    output[out_index++] = input[in_index];
                }
            }
        }
    }
}

void strided_slice_4Dto4D_int32(const int32_t* input, int d1, int d2, int d3, int d4,
                                const int* begin, const int* end, const int* strides,
                                int32_t* output, int o_d1, int o_d2, int o_d3, int o_d4,
                                int begin_mask, int end_mask, int ellipsis_mask,
                                int new_axis_mask, int shrink_axis_mask) {
    auto clamp = [](int v, int lo, int hi) {
        return std::min(std::max(v, lo), hi);
    };

    auto start_for_axis = [&](int axis, int dim_size) {
        int start = begin[axis];
        if (begin_mask & (1 << axis)) {
            start = strides[axis] > 0 ? 0 : dim_size - 1;
        }
        if (start < 0) start += dim_size;
        return clamp(start, strides[axis] > 0 ? 0 : -1, strides[axis] > 0 ? dim_size : dim_size - 1);
    };

    auto stop_for_axis = [&](int axis, int dim_size, int start) {
        int stop = end[axis];
        if (end_mask & (1 << axis)) {
            stop = strides[axis] > 0 ? dim_size : -1;
        }
        if (stop < 0) stop += dim_size;
        return clamp(stop, strides[axis] > 0 ? 0 : -1, strides[axis] > 0 ? dim_size : dim_size - 1);
    };

    int starts[4] = { start_for_axis(0, d1), start_for_axis(1, d2), start_for_axis(2, d3), start_for_axis(3, d4) };
    int stops[4] = { stop_for_axis(0, d1, starts[0]), stop_for_axis(1, d2, starts[1]), stop_for_axis(2, d3, starts[2]), stop_for_axis(3, d4, starts[3]) };

    std::cout << "Starts: " << starts[0] << ", " << starts[1] << ", " << starts[2] << ", " << starts[3] << std::endl;
    std::cout << "Stops: " << stops[0] << ", " << stops[1] << ", " << stops[2] << ", " << stops[3] << std::endl;

    int out_index = 0;
    for (int i = starts[0]; (strides[0] > 0) ? (i < stops[0]) : (i > stops[0]); i += strides[0]) {
        for (int j = starts[1]; (strides[1] > 0) ? (j < stops[1]) : (j > stops[1]); j += strides[1]) {
            for (int k = starts[2]; (strides[2] > 0) ? (k < stops[2]) : (k > stops[2]); k += strides[2]) {
                for (int l = starts[3]; (strides[3] > 0) ? (l < stops[3]) : (l > stops[3]); l += strides[3]) {
                    output[out_index] = input[((i * d2 + j) * d3 + k) * d4 + l];
                    std::cout << "output[" << out_index << "]: " << output[out_index] << std::endl;
                    out_index++;
                }
            }
        }
    }
}


// sub_int8 함수의 정의
void sub_int8(int size, const int8_t* input1, int input1_zero_point, int input1_multiplier, int input1_shift,
              const int8_t* input2, int input2_zero_point, int input2_multiplier, int input2_shift,
              int8_t* output, int output_zero_point, int output_multiplier, int output_shift, int left_shift) {
    for (int i = 0; i < size; ++i) {
        // 입력값을 실수 범위로 변환
        int32_t input1_val = (input1[i] - input1_zero_point) << left_shift;
        int32_t input2_val = (input2[i] - input2_zero_point) << left_shift;

        // 입력값에 승수 및 시프트 적용
        int32_t input1_scaled = input1_val * input1_multiplier >> input1_shift;
        int32_t input2_scaled = input2_val * input2_multiplier >> input2_shift;

        // 뺄셈 수행
        int32_t raw_output = input1_scaled - input2_scaled;

        // 출력 승수 및 시프트 적용
        int32_t scaled_output = raw_output * output_multiplier >> output_shift;

        // 제로 포인트 추가 및 클램핑
        int32_t final_output = scaled_output + output_zero_point;
        final_output = std::min(127, std::max(-128, final_output));

        // 결과를 출력 배열에 저장
        output[i] = static_cast<int8_t>(final_output);
    }
}

// tile_3D_int8 함수 정의
void tile_3D_int8(const int8_t* input, int input_h, int input_w, int input_c,
                  int8_t* output, int output_h, int output_w, int output_c) {
    int input_size = input_h * input_w * input_c;
    int output_size = output_h * output_w * output_c;

    // 반복 횟수 계산
    int rep_h = output_h / input_h;
    int rep_w = output_w / input_w;
    int rep_c = output_c / input_c;

    for (int oh = 0; oh < output_h; ++oh) {
        for (int ow = 0; ow < output_w; ++ow) {
            for (int oc = 0; oc < output_c; ++oc) {
                // 입력 인덱스 계산
                int ih = oh % input_h;
                int iw = ow % input_w;
                int ic = oc % input_c;

                // 입력 및 출력 인덱스 계산
                int input_index = (ih * input_w + iw) * input_c + ic;
                int output_index = (oh * output_w + ow) * output_c + oc;

                // 값 복사
                output[output_index] = input[input_index];
            }
        }
    }
}

// void quantize(const uint8_t* input, int8_t* output, float scale, int zero_point, int size ) {
//     for (int i = 0; i < size; ++i) {
//         // int32_t quantized = std::round((input[i] - zero_point) * scale);
//         // output[i] = std::max(0, std::min(255, quantized));
//         int32_t quantized = std::round(input[i] / scale) + zero_point;
//         output[i] = std::max(-128, std::min(127, quantized));
//     }
// }

void quantize(const uint8_t* input_buffer, int8_t* output_buffer, float scale, int zero_point, int buffer_size) {
    for (int i = 0; i < buffer_size; ++i) {
        // Apply the quantization formula
        float dequantized_value = static_cast<float>(input_buffer[i]) - static_cast<float>(zero_point);
        int32_t quantized_value = static_cast<int32_t>(std::round(dequantized_value / scale));

        // Clip the value to int8 range [-128, 127]
        quantized_value = std::max(-128, std::min(127, quantized_value));

        // Store the quantized value in the output buffer
        output_buffer[i] = static_cast<int8_t>(quantized_value);
    }
}

void quantize(const int8_t* input, uint8_t* output, float scale, int zero_point, int size ) {
    for (int i = 0; i < size; ++i) {
        int32_t quantized = std::round((input[i] - zero_point) * scale);
        output[i] = std::max(0, std::min(255, quantized));
    }
}
