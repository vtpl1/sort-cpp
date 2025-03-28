cmake_minimum_required(VERSION 3.17)
set(VERSION 0.1.0)

project(sort-cpp
  VERSION ${VERSION}
  DESCRIPTION "sort-cpp"
  HOMEPAGE_URL "https://github.com/vtpl1/sort-cpp"
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set(CMAKE_CXX_CLANG_TIDY
#     clang-tidy;)



add_subdirectory(sort-cpp)
