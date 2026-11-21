cmake_minimum_required(VERSION 3.16)
project(TensorC VERSION 0.15.0 LANGUAGES C)

option(TENSORC_BUILD_EXAMPLES "Build TensorC examples" ON)
option(TENSORC_BUILD_BENCHMARKS "Build TensorC benchmarks" ON)
option(TENSORC_USE_ACCELERATE "Use Apple Accelerate for large matrix operations" ON)
option(TENSORC_USE_CUDA "Use NVIDIA CUDA backend for large tensor operations" OFF)

add_library(tensorc minitf.c ndtensor.c rl.c)
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
if(TENSORC_USE_CUDA)
    # Tercer backend tras el mismo switch: ndtensor.c elige por tamaño y
    # cae al camino genérico ante cualquier fallo. Requiere CUDA Toolkit
    # (nvcc) y CMake >= 3.18 para find_package(CUDAToolkit).
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_sources(tensorc PRIVATE tensorc_cuda.cu)
    target_compile_definitions(tensorc PRIVATE TENSORC_HAS_CUDA=1)
    target_link_libraries(tensorc PUBLIC CUDA::cudart CUDA::cublas)
    set_target_properties(tensorc PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED ON)
    message(STATUS "TensorC: backend CUDA activado (TENSORC_HAS_CUDA).")
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
    add_executable(tensorc_cnn_mnist examples/cnn_mnist.c)
    target_link_libraries(tensorc_cnn_mnist PRIVATE TensorC::tensorc)
    add_executable(tensorc_study_classifier examples/study_classifier.c)
    target_link_libraries(tensorc_study_classifier PRIVATE TensorC::tensorc)
    add_executable(tensorc_dqn_cartpole examples/dqn_cartpole.c)
    target_link_libraries(tensorc_dqn_cartpole PRIVATE TensorC::tensorc)
    find_package(JPEG QUIET)
    if(JPEG_FOUND)
        add_executable(tensorc_dqn_study_bandit
            experiments/study_classifier/dqn_study_bandit.c)
        target_link_libraries(tensorc_dqn_study_bandit PRIVATE
            TensorC::tensorc JPEG::JPEG)
    else()
        message(STATUS "libjpeg not found; skipping tensorc_dqn_study_bandit")
    endif()
endif()

if(TENSORC_BUILD_BENCHMARKS)
    add_executable(tensorc_conv2d_benchmark benchmarks/conv2d/conv2d_benchmark.c)
    target_link_libraries(tensorc_conv2d_benchmark PRIVATE TensorC::tensorc)
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
    add_executable(tensorc_test_nd_serialization tests/test_nd_serialization.c)
    target_link_libraries(tensorc_test_nd_serialization PRIVATE TensorC::tensorc)
    add_test(NAME tensorc_nd_serialization COMMAND tensorc_test_nd_serialization)
    add_executable(tensorc_test_conv2d tests/test_conv2d.c)
    target_link_libraries(tensorc_test_conv2d PRIVATE TensorC::tensorc)
    add_test(NAME tensorc_conv2d COMMAND tensorc_test_conv2d)
    add_executable(tensorc_test_conv2d_layer tests/test_conv2d_layer.c)
    target_link_libraries(tensorc_test_conv2d_layer PRIVATE TensorC::tensorc)
    add_test(NAME tensorc_conv2d_layer COMMAND tensorc_test_conv2d_layer)
    add_executable(tensorc_test_optimizers tests/test_optimizers.c)
    target_link_libraries(tensorc_test_optimizers PRIVATE TensorC::tensorc)
    add_test(NAME tensorc_optimizers COMMAND tensorc_test_optimizers)
    add_executable(tensorc_test_dense_layer tests/test_dense_layer.c)
    target_link_libraries(tensorc_test_dense_layer PRIVATE TensorC::tensorc)
    add_test(NAME tensorc_dense_layer COMMAND tensorc_test_dense_layer)
    add_executable(tensorc_test_recurrent tests/test_recurrent.c)
    target_link_libraries(tensorc_test_recurrent PRIVATE TensorC::tensorc)
    add_test(NAME tensorc_recurrent COMMAND tensorc_test_recurrent)
    add_executable(tensorc_test_transformer tests/test_transformer.c)
    target_link_libraries(tensorc_test_transformer PRIVATE TensorC::tensorc)
    add_test(NAME tensorc_transformer COMMAND tensorc_test_transformer)
    add_executable(tensorc_test_language_ops tests/test_language_ops.c)
    target_link_libraries(tensorc_test_language_ops PRIVATE TensorC::tensorc)
    add_test(NAME tensorc_language_ops COMMAND tensorc_test_language_ops)
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
install(FILES include/tensorc/rl.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tensorc)
install(EXPORT TensorCTargets NAMESPACE TensorC:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TensorC)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/TensorCConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/TensorCConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TensorC
)
