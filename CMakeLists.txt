cmake_minimum_required(VERSION 3.16)
project(TensorC VERSION 0.5.0 LANGUAGES C)

option(TENSORC_BUILD_EXAMPLES "Build TensorC examples" ON)
option(TENSORC_USE_ACCELERATE "Use Apple Accelerate for large matrix operations" ON)

add_library(tensorc minitf.c ndtensor.c)
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
if(APPLE AND TENSORC_USE_ACCELERATE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if(ACCELERATE_FRAMEWORK)
        target_compile_definitions(tensorc PRIVATE TENSORC_HAS_ACCELERATE=1 ACCELERATE_NEW_LAPACK=1)
        target_link_libraries(tensorc PUBLIC ${ACCELERATE_FRAMEWORK})
    else()
        message(WARNING "Apple Accelerate no está disponible; se usará el backend genérico.")
    endif()
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
    add_executable(tensorc_autograd examples/autograd.c)
    target_link_libraries(tensorc_autograd PRIVATE TensorC::tensorc)
    add_executable(tensorc_fahrenheit examples/fahrenheit.c)
    target_link_libraries(tensorc_fahrenheit PRIVATE TensorC::tensorc)
    add_executable(tensorc_matmul_benchmark examples/matmul_benchmark.c)
    target_link_libraries(tensorc_matmul_benchmark PRIVATE TensorC::tensorc)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(tensorc_test_autograd tests/test_autograd.c)
    target_link_libraries(tensorc_test_autograd PRIVATE TensorC::tensorc)
    add_test(NAME tensorc_autograd COMMAND tensorc_test_autograd)
    add_executable(tensorc_test_matmul tests/test_matmul.c)
    target_link_libraries(tensorc_test_matmul PRIVATE TensorC::tensorc)
    add_test(NAME tensorc_matmul COMMAND tensorc_test_matmul)
    add_executable(tensorc_test_broadcast tests/test_broadcast.c)
    target_link_libraries(tensorc_test_broadcast PRIVATE TensorC::tensorc)
    add_test(NAME tensorc_broadcast COMMAND tensorc_test_broadcast)
    add_executable(tensorc_test_serialization tests/test_serialization.c)
    target_link_libraries(tensorc_test_serialization PRIVATE TensorC::tensorc)
    add_test(NAME tensorc_serialization COMMAND tensorc_test_serialization)
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
install(FILES include/tensorc/ndtensor.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tensorc)
install(EXPORT TensorCTargets NAMESPACE TensorC:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TensorC)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/TensorCConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/TensorCConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TensorC
)
