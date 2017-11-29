# Copyright (C) 2016 Ammar Husain. All Rights Reserved.
cmake_minimum_required(VERSION 3.3)

#set(CMAKE_CXX_FLAGS "-Wall -g")

#project(algorithm-sandbox)

# find_package(Protobuf REQUIRED)
#find_package(Eigen3 REQUIRED)

add_executable(tflow-cpp loader_est.cpp)

message("trying to find tensorflow")
find_package(TensorflowCC COMPONENTS Static)

enable_testing()

set(CMAKE_CXX_FLAGS "-Wall -g -std=c++11")

include(FindProtobuf.cmake)


target_link_libraries(tflow-cpp TensorflowCC::Static ${PROTOBUF_LIBRARIES})

target_include_directories(tflow-cpp PUBLIC ${PROTOBUF_INCLUDE_DIRS})

message("pb: ${PROTOBUF_LIBRARIES}")