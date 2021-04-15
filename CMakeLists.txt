cmake_minimum_required(VERSION 3.19)
project(nn)

set(CMAKE_CXX_STANDARD 20)

add_executable(nn MLP.cpp NeuralNetworks.cpp)