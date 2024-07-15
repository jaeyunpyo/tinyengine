#include <algorithm>
#include <cstdint>
#include "arm_math.h"  // CMSIS-DSP 라이브러리

// 함수 정의
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
    for (int c = 0; c < input_c; ++c) {
        for (int h = 0; h < input_h; ++h) {
            for (int w = 0; w < input_w; ++w) {
                sbuf[(c * padded_input_h + (h + pad)) * padded_input_w + (w + pad)] = input[c * input_h * input_w + h * input_w + w];
            }
        }
    }

    // 16x16 패치 Conv2D 연산을 수행합니다.
    int kernel_size = 16;
    int patch_size = kernel_size * kernel_size;

    for (int oc = 0; oc < output_c; ++oc) {
        for (int oh = 0; oh < output_h; ++oh) {
            for (int ow = 0; ow < output_w; ++ow) {
                int32_t acc = bias[oc];

                for (int ic = 0; ic < input_c; ++ic) {
                    int16_t input_patch[patch_size];
                    int16_t weight_patch[patch_size];
                    int idx = 0;

                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int h_in = oh * stride + kh;
                            int w_in = ow * stride + kw;

                            if (h_in >= 0 && h_in < padded_input_h && w_in >= 0 && w_in < padded_input_w) {
                                input_patch[idx] = sbuf[(ic * padded_input_h + h_in) * padded_input_w + w_in] - input_zero_point;
                            } else {
                                input_patch[idx] = 0;
                            }
                            weight_patch[idx] = weights[(oc * input_c + ic) * patch_size + kh * kernel_size + kw];
                            idx++;
                        }
                    }

                    int32_t dot_product;
                    arm_dot_prod_q7((q7_t*)input_patch, (q7_t*)weight_patch, patch_size, &dot_product);
                    acc += dot_product;
                }

                // Apply the scale and zero points, then clamp the result
                float scaled = acc * scales[oc];
                int32_t sum = static_cast<int32_t>(scaled);
                sum += output_zero_point;
                sum = std::max(min_val, std::min(max_val, sum));
                output[(oc * output_h + oh) * output_w + ow] = static_cast<int8_t>(sum);
            }
        }
    }
}
