#include "genModel.h"
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <vector>

#define IMAGE_WIDTH 224
#define IMAGE_HEIGHT 224
#define IMAGE_CHANNELS 3

#define INPUT_SIZE 150528  // 3 * 224 * 224
#define OUTPUT_SIZE 1000

// 입력 데이터와 출력 데이터를 위한 배열을 생성합니다.
unsigned char random_buffer[INPUT_SIZE];
unsigned char output[OUTPUT_SIZE];

int main() {

    std::ifstream file("test_image.bin", std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file" << std::endl;
        return 1;
    }

    //  signed char random_buffer[INPUT_SIZE];
    file.read(reinterpret_cast<char*>(random_buffer), INPUT_SIZE);
    if (!file) {
        std::cerr << "Failed to read image data" << std::endl;
        return 1;
    }

    file.close();

    // 모델을 호출하여 추론을 수행합니다.
    genModel(random_buffer, output);

    printf("\n\ninput\n");
    // input 출력 100개씩 10줄 출력하기
    for(int i = 0; i < 1000; i++) {
        printf("%02x ", static_cast<unsigned char>(random_buffer[i]));
        if(i % 100 == 99) std::cout << std::endl;
    }

    printf("output\n");
    // 결과 출력 또는 추가 작업
    // 한 줄개 100개씩 10줄 출력하기
    for(int i = 0; i < 1000; i++) {
        printf("%02x ", static_cast<unsigned char>(output[i]));
        if(i % 100 == 99) std::cout << std::endl;
    }

    return 0;
}
