cmake_minimum_required(VERSION 3.17)

project(sort-cpp LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_CLANG_TIDY
    clang-tidy;)

add_subdirectory(sort-c++)
