cmake_minimum_required(VERSION 3.16)
project(TensorC VERSION 0.1.0 LANGUAGES C)

option(TENSORC_BUILD_EXAMPLES "Build TensorC examples" ON)

add_library(tensorc minitf.c)
add_library(TensorC::tensorc ALIAS tensorc)
target_compile_features(tensorc PUBLIC c_std_11)
target_include_directories(tensorc
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)
if(MSVC)
    target_compile_options(tensorc PRIVATE /W4)
else()
    target_compile_options(tensorc PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(tensorc PUBLIC m)
endif()
set_target_properties(tensorc PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
    PUBLIC_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/include/tensorc/tensorc.h"
)

if(TENSORC_BUILD_EXAMPLES)
    add_executable(tensorc_xor examples/xor_public_api.c)
    target_link_libraries(tensorc_xor PRIVATE TensorC::tensorc)
    add_executable(tensorc_xor_adam examples/xor_adam.c)
    target_link_libraries(tensorc_xor_adam PRIVATE TensorC::tensorc)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
configure_package_config_file(
    cmake/TensorCConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/TensorCConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TensorC
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/TensorCConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(TARGETS tensorc EXPORT TensorCTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tensorc
)
install(FILES minitf.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT TensorCTargets NAMESPACE TensorC:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TensorC)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/TensorCConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/TensorCConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TensorC
)
