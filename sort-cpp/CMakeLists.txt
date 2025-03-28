find_package(OpenCV REQUIRED)

include(GNUInstallDirs)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

add_library(sort_cpp
    Hungarian.cpp
    KalmanTracker.cpp
    SortTracker.cpp
)

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

include(GenerateExportHeader)
generate_export_header(sort_cpp)

list(APPEND sort_cpp_HEADERS
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/SortTracker.h>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/KalmanTracker.h>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/sort_cpp_export.h>
)

target_include_directories(sort_cpp
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/sort_cpp>
)


target_link_libraries(sort_cpp
    PUBLIC opencv_video
)

set_target_properties(sort_cpp
    PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

set_target_properties(sort_cpp
    PROPERTIES
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION ${PROJECT_VERSION}
)

set(INSTALL_TARGETS
    sort_cpp
)

include(CMakePackageConfigHelpers)


configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/sort_cpp-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_DATADIR}/sort_cpp
    NO_SET_AND_CHECK_MACRO
)

write_basic_package_version_file(
    sort_cpp-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(TARGETS ${INSTALL_TARGETS}
    EXPORT sort_cpp-targets

    # on Windows put the dlls into bin
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/$<CONFIG>
    COMPONENT sort_cpp_Runtime

    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/$<CONFIG>
    COMPONENT sort_cpp_Runtime
    NAMELINK_COMPONENT sort_cpp_Development

    # ... and the import lib into the devel package
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/$<CONFIG>
    COMPONENT sort_cpp_Development

    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/sort_cpp
    COMPONENT sort_cpp_Development
)

install(EXPORT sort_cpp-targets
    DESTINATION ${CMAKE_INSTALL_DATADIR}/sort_cpp
    NAMESPACE sort_cpp::
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/sort_cpp-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/sort_cpp-config-version.cmake
    DESTINATION
        ${CMAKE_INSTALL_DATADIR}/sort_cpp
    COMPONENT sort_cpp
)

install(
    FILES
        ${sort_cpp_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/sort_cpp
    COMPONENT sort_cpp
)

add_library(sort_cpp::sort_cpp ALIAS sort_cpp)