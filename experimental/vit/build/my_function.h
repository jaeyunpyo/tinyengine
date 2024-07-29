#include <cstring>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric> // For std::iota

using namespace std;

// 양자화된 텐서의 덧셈을 수행하는 함수
void add_fpreq(int length, uint8_t* input1, float input1_scale, int input1_zero_point,
               uint8_t* input2, float input2_scale, int input2_zero_point,
               float output_scale, int output_zero_point, uint8_t* output) {
    for (int i = 0; i < length; ++i) {
        // 입력 텐서를 실수값으로 변환
        float real_input1 = (input1[i] - input1_zero_point) * input1_scale;
        float real_input2 = (input2[i] - input2_zero_point) * input2_scale;

        // 두 입력 텐서를 더함
        float real_output = real_input1 + real_input2;

        // 결과를 양자화된 값으로 변환
        int32_t quantized_output = static_cast<int32_t>(std::round(real_output / output_scale)) + output_zero_point;

        // 출력 범위를 -128에서 127 사이로 제한
        output[i] = static_cast<int8_t>(std::max(-128, std::min(127, quantized_output)));
    }
}

// avg_pooling 함수의 정의
void avg_pooling(uint8_t* input, int input_h, int input_w, int input_c,
                 int filter_h, int filter_w, int stride_h, int stride_w,
                 int pad_h, int pad_w,
                 uint8_t* output) {
    // 패딩을 포함한 입력 크기 계산
    int padded_input_h = input_h + 2 * pad_h;
    int padded_input_w = input_w + 2 * pad_w;

    // 출력 크기 계산
    int output_h = (padded_input_h - filter_h) / stride_h + 1;
    int output_w = (padded_input_w - filter_w) / stride_w + 1;

    // 입력 텐서를 순회하면서 평균 풀링 연산 수행
    for (int h = 0; h < output_h; ++h) {
        for (int w = 0; w < output_w; ++w) {
            for (int c = 0; c < input_c; ++c) {
                int32_t sum = 0;
                for (int fh = 0; fh < filter_h; ++fh) {
                    for (int fw = 0; fw < filter_w; ++fw) {
                        int ih = h * stride_h + fh - pad_h;
                        int iw = w * stride_w + fw - pad_w;
                        if (ih < 0 || ih >= input_h || iw < 0 || iw >= input_w) {
                            // 패딩 값 사용 (0으로 가정)
                            sum += 0;
                        } else {
                            sum += input[(ih * input_w + iw) * input_c + c];
                        }
                    }
                }
                // 평균 값 계산
                int filter_area = filter_h * filter_w;
                int32_t avg = sum / filter_area;
                // 값 제한 (클램핑)
                avg = std::max(-128, std::min(127, avg));
                // 출력에 저장
                output[(h * output_w + w) * input_c + c] = static_cast<int8_t>(avg);
            }
        }
    }
}

// batch_matmul 함수의 정의
void batch_matmul(int8_t* input, int8_t* input2, int8_t* output, 
                  int batch_size, int M, int K, int N, 
                  bool adj_x, bool adj_y,
                  int input_zero_point, int input2_zero_point, int output_zero_point,
                  float input_scale, float input2_scale, float output_scale,
                  int input_shift, int input2_shift, int output_shift,
                  int input_multiplier, int input2_multiplier, int output_multiplier) {
    for (int b = 0; b < batch_size; ++b) {
        int8_t* A = input + b * M * K;
        int8_t* B = input2 + b * K * N;
        int8_t* C = output + b * M * N;

        // Transpose A if adj_x is true
        int8_t* tempA = nullptr;
        if (adj_x) {
            tempA = new int8_t[M * K];
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < K; ++j) {
                    tempA[j * M + i] = A[i * K + j];
                }
            }
            A = tempA;
            std::swap(M, K);
        }

        // Transpose B if adj_y is true
        int8_t* tempB = nullptr;
        if (adj_y) {
            tempB = new int8_t[K * N];
            for (int i = 0; i < K; ++i) {
                for (int j = 0; j < N; ++j) {
                    tempB[j * K + i] = B[i * N + j];
                }
            }
            B = tempB;
            std::swap(K, N);
        }

        // Initialize C to zero
        std::memset(C, 0, sizeof(int8_t) * M * N);

        // Perform matrix multiplication
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                int32_t sum = 0; // Use int32_t to accumulate results to avoid overflow
                for (int k = 0; k < K; ++k) {
                    int32_t a_val = A[i * K + k] - input_zero_point;
                    int32_t b_val = B[k * N + j] - input2_zero_point;
                    sum += a_val * b_val;
                }
                // Quantize the accumulated result back to int8_t
                float scaled_sum = sum * input_scale * input2_scale / output_scale;
                int32_t quantized_output = static_cast<int32_t>(round(scaled_sum)) + output_zero_point;
                quantized_output = std::min(std::max(quantized_output, -128), 127);
                C[i * N + j] = static_cast<int8_t>(quantized_output);
            }
        }

        // Clean up temporary arrays
        if (adj_x) {
            delete[] A;
        }
        if (adj_y) {
            delete[] B;
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
void conv2d_16x16_fpreq(const uint8_t* input, int input_w, int input_h, int input_c,
                        const int8_t* weights, const int32_t* bias, const float* scales,
                        int output_w, int output_h, int output_c,
                        uint8_t* output, int stride, int pad, 
                        int input_zero_point, int output_zero_point, int min_val, int max_val, 
                        int16_t* sbuf)
{
    // 패딩을 적용한 입력 크기를 계산합니다.
    int padded_input_w = input_w + 2 * pad;
    int padded_input_h = input_h + 2 * pad;

    // 임시 버퍼를 초기화합니다.
    std::fill(sbuf, sbuf + padded_input_w * padded_input_h * input_c, input_zero_point);

    // 패딩을 적용하여 입력을 임시 버퍼에 복사합니다.
    for (int c = 0; c < input_c; ++c) {
        for (int h = 0; h < input_h; ++h) {
            for (int w = 0; w < input_w; ++w) {
                sbuf[(c * padded_input_h + (h + pad)) * padded_input_w + (w + pad)] = input[c * input_h * input_w + h * input_w + w];
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
    // 출력을 0으로 초기화
    std::memset(output, 0, sizeof(int8_t) * output_size);

    for (int out_idx = 0; out_idx < output_size; ++out_idx) {
        int32_t acc = (bias != nullptr) ? bias[out_idx] : 0;

        for (int in_idx = 0; in_idx < input_size; ++in_idx) {
            int32_t input_val = input[in_idx] - input_zero_point;
            int32_t weight_val = weights[out_idx * input_size + in_idx] - weight_zero_point;
            acc += input_val * weight_val;
        }

        // 양자화된 값으로 변환
        float scaled_acc = acc * input_scale * weight_scale / output_scale;
        int32_t quantized_output = static_cast<int32_t>(round(scaled_acc)) + output_zero_point;

        // 값 제한 (클램핑)
        if (activation == 1) { // Assuming 1 represents ReLU
            quantized_output = std::max(0, quantized_output);
        }

        quantized_output = std::min(127, std::max(-128, quantized_output));
        output[out_idx] = static_cast<int8_t>(quantized_output);

        // printf("acc: %d, sca_acc: %0.2lf, out: %d\n", acc, scaled_acc, quantized_output);
    }
}

// gather 함수의 정의
void gather(const uint8_t* input, const uint8_t* indices, uint8_t* output, int num_indices, int input_size) {
    if (indices == nullptr) {
        // indices 값이 Null인 경우 입력 배열을 그대로 출력 배열로 복사
        for (int i = 0; i < input_size; ++i) {
            output[i] = input[i];
        }
    } else {
        // indices 값이 유효한 경우 기존 gather 동작 수행
        for (int i = 0; i < num_indices; ++i) {
            int idx = indices[i];
            if (idx < 0 || idx >= input_size) {
                // 인덱스가 유효하지 않은 경우 0을 반환하거나 다른 오류 처리를 할 수 있음
                output[i] = 0; // 또는 적절한 오류 처리를 추가
            } else {
                output[i] = input[idx];
            }
        }
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

// rsqrt 함수의 정의 
void rsqrt(const int8_t* input, int8_t* output, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int index = i * cols + j;
            output[index] = 1.0f / std::sqrt(input[index]);
        }
    }
}

// shape 함수의 정의
void shape(const int* input_tensor, int num_dimensions, int8_t* output_tensor) {
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
        // 입력 텐서와 출력 텐서의 차원 수 계산
        int input_dims[4] = {d1, d2, d3, d4};
        int output_dims[4] = {o_d1, o_d2, o_d3, o_d4};

        // 슬라이싱 루프
        for (int i = 0; i < o_d1; ++i) {
            for (int j = 0; j < o_d2; ++j) {
                for (int k = 0; k < o_d3; ++k) {
                    for (int l = 0; l < o_d4; ++l) {
                        // 계산된 인덱스
                        int in_i = begin[0] + i * strides[0];
                        int in_j = begin[1] + j * strides[1];
                        int in_k = begin[2] + k * strides[2];
                        int in_l = begin[3] + l * strides[3];

                        // 범위 검사
                        if (in_i >= end[0] || in_j >= end[1] || in_k >= end[2] || in_l >= end[3]) {
                            continue;
                        }

                        // 1D 인덱스로 변환
                        int input_index = ((in_i * input_dims[1] + in_j) * input_dims[2] + in_k) * input_dims[3] + in_l;
                        int output_index = ((i * output_dims[1] + j) * output_dims[2] + k) * output_dims[3] + l;

                        // 출력 텐서에 값 복사
                        output[output_index] = input[input_index];
                    
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
