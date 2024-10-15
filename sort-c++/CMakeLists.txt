cmake_minimum_required(VERSION 3.17)

find_package(OpenCV REQUIRED)

add_library(sort_cpp STATIC
  Hungarian.cpp
  KalmanTracker.cpp
  SortTracker.cpp
)

target_precompile_headers(sort_cpp
  PUBLIC inc/Hungarian.h
  PUBLIC inc/KalmanTracker.h
  PUBLIC inc/SortTracker.h
)

target_include_directories(sort_cpp
  PUBLIC inc
)

target_link_libraries(sort_cpp
  PUBLIC opencv_video
)

add_library(sort_cpp::sort_cpp ALIAS sort_cpp)

# set_property(TARGET sort_cpp PROPERTY
#   MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

add_executable(sort_cpp_exe
  main.cpp
)

target_link_libraries(sort_cpp_exe
  PRIVATE sort_cpp
)

# set_property(TARGET sort_cpp_exe PROPERTY
#   MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")