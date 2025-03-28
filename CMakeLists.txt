cmake_minimum_required(VERSION 3.17)
set(VERSION 0.2.0)

project(sortcpp
  VERSION ${VERSION}
  DESCRIPTION "sortcpp"
  HOMEPAGE_URL "https://github.com/vtpl1/sortcpp"
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set(CMAKE_CXX_CLANG_TIDY
#     clang-tidy;)



add_subdirectory(sortcpp)
