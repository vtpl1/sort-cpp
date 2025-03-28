find_package(OpenCV REQUIRED)

include(GNUInstallDirs)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

add_library(sortcpp
    Hungarian.cpp
    KalmanTracker.cpp
    SortTracker.cpp
)

# set_property(TARGET sortcpp PROPERTY
#   MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

add_executable(sortcpp_exe
    main.cpp
)

target_link_libraries(sortcpp_exe
    PRIVATE sortcpp
)

# set_property(TARGET sortcpp_exe PROPERTY
#   MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

include(GenerateExportHeader)
generate_export_header(sortcpp)

list(APPEND sortcpp_HEADERS
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/SortTracker.h>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/KalmanTracker.h>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/sortcpp_export.h>
)

target_include_directories(sortcpp
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/sortcpp>
)


target_link_libraries(sortcpp
    PUBLIC opencv_video
)

set_target_properties(sortcpp
    PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

set_target_properties(sortcpp
    PROPERTIES
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION ${PROJECT_VERSION}
)

set(INSTALL_TARGETS
    sortcpp
)

include(CMakePackageConfigHelpers)


configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/sortcpp-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_DATADIR}/sortcpp
    NO_SET_AND_CHECK_MACRO
)

write_basic_package_version_file(
    sortcpp-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(TARGETS ${INSTALL_TARGETS}
    EXPORT sortcpp-targets

    # on Windows put the dlls into bin
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/$<CONFIG>
    COMPONENT sortcpp_Runtime

    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/$<CONFIG>
    COMPONENT sortcpp_Runtime
    NAMELINK_COMPONENT sortcpp_Development

    # ... and the import lib into the devel package
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/$<CONFIG>
    COMPONENT sortcpp_Development

    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/sortcpp
    COMPONENT sortcpp_Development
)

install(EXPORT sortcpp-targets
    DESTINATION ${CMAKE_INSTALL_DATADIR}/sortcpp
    NAMESPACE sortcpp::
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/sortcpp-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/sortcpp-config-version.cmake
    DESTINATION
        ${CMAKE_INSTALL_DATADIR}/sortcpp
    COMPONENT sortcpp
)

install(
    FILES
        ${sortcpp_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/sortcpp
    COMPONENT sortcpp
)

add_library(sortcpp::sortcpp ALIAS sortcpp)