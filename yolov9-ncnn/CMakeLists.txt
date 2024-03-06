cmake_minimum_required(VERSION 3.27)
set(PROJECT_NAME yolov9)
set(SRC yolov9.cpp)

project(${PROJECT_NAME})
find_package(OpenCV REQUIRED)
include_directories(
    include/ncnn
    ${OpenCV_INCLUDE_DIRS}
)

link_directories(lib)

add_executable(${PROJECT_NAME} ${SRC})
target_link_libraries(${PROJECT_NAME} PUBLIC 
    ${OpenCV_LIBS}
    libncnn.a)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()


