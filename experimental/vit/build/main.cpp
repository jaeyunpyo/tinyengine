#include "genModel.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

#define INPUT_SIZE 150528  // 3 * 224 * 224
#define OUTPUT_SIZE 1000

// 입력 데이터와 출력 데이터를 위한 배열을 생성합니다.
signed char random_buffer[INPUT_SIZE];
signed char output[OUTPUT_SIZE];

int main() {
    
    // 랜덤 입력 데이터를 생성합니다.
    std::srand(std::time(0));  // 랜덤 시드 설정
    for (int i = 0; i < INPUT_SIZE; ++i) {
        random_buffer[i] = std::rand() % 256;  // -128 ~ 127 범위의 랜덤 값
    }

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
