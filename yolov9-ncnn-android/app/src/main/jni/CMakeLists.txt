project(yolov9ncnn)

cmake_minimum_required(VERSION 3.4.1)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20230816-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(yolov9ncnn SHARED yolov9ncnn_jni.cpp)

target_link_libraries(yolov9ncnn
    ncnn

    jnigraphics
)
