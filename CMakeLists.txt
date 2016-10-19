project(LeNet_5)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -lm -mavx -fopenmp")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -lm -mavx -fopenmp")

set(SOURCE_FILES main.c batch.c predict.c lenet.h)
add_executable(LeNet_5 ${SOURCE_FILES})