cmake_minimum_required(VERSION 3.10)
project(TFLiteExample)

set(CMAKE_CXX_STANDARD 17) # C++17로 설정

set(TENSORFLOW_LITE_DIR /mnt/d/workspace/tensorflow)
set(FLATBUFFERS_INCLUDE_DIR /mnt/d/workspace/tinyengine/experimental/vit/tflite/flatbuffers/include) # flatbuffers 포함 경로 설정

include_directories(${TENSORFLOW_LITE_DIR})
include_directories(${FLATBUFFERS_INCLUDE_DIR}) # flatbuffers 포함 경로 추가
link_directories(${TENSORFLOW_LITE_DIR}/bazel-bin/tensorflow/lite)

add_executable(TFLiteExample tflite.cpp)
target_link_libraries(TFLiteExample tensorflowlite)
