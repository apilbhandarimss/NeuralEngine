cmake_minimum_required(VERSION 3.20)
project(NeuralEngine VERSION 0.1.0 LANGUAGES CXX)

option(NEURAL_BUILD_TESTS "Build the test suite" ON)
option(NEURAL_BUILD_EXAMPLES "Build examples" ON)

add_library(neural_engine
    src/core/device.cpp
    src/tensor/tensor.cpp
    src/backend/backend.cpp
    src/ops/activations.cpp
    src/ops/arithmetic.cpp
    src/ops/loss.cpp
    src/nn/layer.cpp
    src/nn/model.cpp
    src/optim/optimizer.cpp
    src/data/dataset.cpp
)
target_compile_features(neural_engine PUBLIC cxx_std_20)
target_include_directories(neural_engine
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
set_target_properties(neural_engine PROPERTIES
    CXX_EXTENSIONS OFF
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

if(MSVC)
    target_compile_options(neural_engine PRIVATE /W4 /permissive-)
else()
    target_compile_options(neural_engine PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()

if(NEURAL_BUILD_TESTS)
    enable_testing()
    add_executable(neural_engine_tests tests/test_main.cpp)
    target_link_libraries(neural_engine_tests PRIVATE neural_engine)
    add_test(NAME neural_engine_tests COMMAND neural_engine_tests)
    add_executable(neural_gradient_tests tests/test_gradients.cpp)
target_link_libraries(neural_gradient_tests PRIVATE neural_engine)
add_test(NAME neural_gradient_tests COMMAND neural_gradient_tests)
endif()

if(NEURAL_BUILD_EXAMPLES)
    add_executable(basic_example examples/basic.cpp)
    target_link_libraries(basic_example PRIVATE neural_engine)
endif()
