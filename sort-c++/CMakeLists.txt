cmake_minimum_required(VERSION 3.17)

find_package( OpenCV REQUIRED )

add_executable(sort_cpp
    Hungarian.cpp
    KalmanTracker.cpp
    main.cpp
)

target_link_libraries(sort_cpp
    PUBLIC ${OpenCV_LIBS}
)

set_property(TARGET sort_cpp PROPERTY
  MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
