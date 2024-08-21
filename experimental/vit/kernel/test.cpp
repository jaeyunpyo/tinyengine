#include <iostream>
#include <fstream>
#include <cstdint>
#include <cstring>

#define IMAGE_WIDTH 224
#define IMAGE_HEIGHT 224
#define IMAGE_CHANNELS 3

#define INPUT_SIZE (IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS) // 3 * 224 * 224
#define BUFFER_SIZE 5447360 // 필요한 전체 버퍼 크기
#define STRIDED_SLICE_OUTPUT_SIZE (IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS) // Strided Slice 출력 크기

// 입력 데이터와 출력 데이터를 위한 배열을 생성합니다.
uint8_t random_buffer[INPUT_SIZE];

// 큰 1D 버퍼를 생성하여 필요한 영역을 재활용합니다.
static uint8_t buffer[BUFFER_SIZE];
static uint8_t *buffer0 = &buffer[0];

// 입력 데이터를 위한 포인터를 반환하는 함수
uint8_t* getInput() {
    return &buffer0[226376];
}

// int8_t 타입을 위한 strided slice 함수
void strided_slice_4Dto4D_int32(const int8_t* input, int d1, int d2, int d3, int d4,
                                const int* begin, const int* end, const int* strides,
                                int8_t* output, int o_d1, int o_d2, int o_d3, int o_d4,
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

                    // 범위 검사 (end 값이 0인 경우를 처리)
                    if ((end[0] != 0 && in_i >= end[0]) || 
                        (end[1] != 0 && in_j >= end[1]) || 
                        (end[2] != 0 && in_k >= end[2]) || 
                        (end[3] != 0 && in_l >= end[3])) {
                        continue;
                    }

                    // 1D 인덱스로 변환
                    int input_index = ((in_i * input_dims[1] + in_j) * input_dims[2] + in_k) * input_dims[3] + in_l;
                    int output_index = ((i * output_dims[1] + j) * output_dims[2] + k) * output_dims[3] + l;

                    // 출력 텐서에 값 복사
                    output[output_index] = input[input_index];
                    printf("[%d] %d \t", output_index, output[output_index]);
                }
                printf("\n");
            }
        }
    }
}

// shape 함수의 정의
void shape(const int* input_tensor, int num_dimensions, int* output_tensor) {
    for (int i = 0; i < num_dimensions; ++i) {
        output_tensor[i] = input_tensor[i];
    }
}

int main() {
    std::ifstream file("../build/test_image.bin", std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file" << std::endl;
        return 1;
    }

    file.read(reinterpret_cast<char*>(random_buffer), INPUT_SIZE);
    if (!file) {
        std::cerr << "Failed to read image data" << std::endl;
        return 1;
    }

    file.close();

    // 읽어온 데이터를 출력합니다.
    for (int i = 0; i < 225; ++i) {
        std::cout << static_cast<int>(random_buffer[i]) << " ";
        if ((i + 1) % 50 == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;

    /* Convert the OpenCV image from BGR to RGB */
    uint8_t* input = getInput();
    uint8_t* input_start = input;  // 변환된 데이터를 저장할 시작 포인터
    int num_row = IMAGE_HEIGHT;
    int num_col = IMAGE_WIDTH;
    int num_channel = IMAGE_CHANNELS;

    for (int i = 0; i < num_row; i++) {
        for (int j = 0; j < num_col; j++) {
            uint8_t b = random_buffer[(i * num_col + j) * num_channel + 0];
            uint8_t g = random_buffer[(i * num_col + j) * num_channel + 1];
            uint8_t r = random_buffer[(i * num_col + j) * num_channel + 2];

            *input++ = (uint8_t)r;
            *input++ = (uint8_t)g;
            *input++ = (uint8_t)b;
        }
    }

    // 변환된 데이터를 출력합니다.
    input = input_start;
    for (int i = 0; i < 225; ++i) {
        std::cout << static_cast<int>(input[i]) << " ";
        if ((i + 1) % 50 == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;

    printf("/* layer 0:SHAPE */\n");
    const int shape_values0[] = {1, 3, 224, 224};
    int32_t shape_buffer[4]; // buffer0 크기 조정
    shape(shape_values0, 4, shape_buffer);
    for (int i = 0; i < 4; i++) {
        printf("%d \t", shape_buffer[i]);
    }
    printf("\n");

    printf("/* layer 1:STRIDED_SLICE */\n");
    const int begin0[] = {0, 0, 0, 0};
    const int end0[] = {1, 0, 0, 0}; // 첫 번째 차원만 슬라이스
    const int strides0[] = {1, 1, 1, 1};

    // 슬라이싱 연산을 위해 출력 버퍼의 크기를 결정
    int8_t output_buffer[INPUT_SIZE]; // 최소 크기로 설정
    
    strided_slice_4Dto4D_int32(
        reinterpret_cast<int8_t*>(input_start),
        4, 1, 1, 1, 
        begin0, end0, strides0, 
        output_buffer, 
        1, 1, 1, 1, 
        0, 0, 0, 0, 1
    );

    for (int i = 0; i < 224; i++) {
        printf("%d \t", output_buffer[i]);
    }
    printf("\n");

    return 0;
}
