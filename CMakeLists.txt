cmake_minimum_required(VERSION 3.28)
project(NeuralNetwork LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Resource path macro
set(NN_RESOURCE_DIR_PATH "${CMAKE_CURRENT_SOURCE_DIR}/resources/fonts")

# Sanitizer flags
add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
add_link_options(-fsanitize=address)

# ------------------------------------------------------------------
# Fetch dependencies
include(FetchContent)

FetchContent_Declare(SFML
    GIT_REPOSITORY https://github.com/SFML/SFML.git
    GIT_TAG 3.0.1
    GIT_SHALLOW ON
    SYSTEM
    EXCLUDE_FROM_ALL
)

FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.12.0
    GIT_SHALLOW ON
    SYSTEM
    EXCLUDE_FROM_ALL
)

FetchContent_MakeAvailable(SFML nlohmann_json)

# ------------------------------------------------------------------
# Main library
add_library(NeuralNetwork STATIC
    src/model/AiModel.cpp
    src/model/model.cpp
    src/model/activations.cpp
    src/model/neuralNetwork.cpp
    src/model/LayerParameters.cpp
    src/model/neuron.cpp
    src/model/layer.cpp
    src/model/config.cpp
    src/trainer/trainer.cpp
    src/trainer/dataBase.cpp
    src/trainer/backPropagation.cpp
    src/trainer/gradient.cpp
    src/trainer/learning_rate.cpp
    src/visualizer/VisualizerRenderer.cpp
    src/visualizer/VisualizerController.cpp
    src/visualizer/visualNN.cpp
    src/visualizer/visualL.cpp
    src/visualizer/state.cpp
    src/visualizer/VInterface.cpp
    src/visualizer/button.cpp
    src/visualizer/fonts.cpp
    src/visualizer/Vstatus.cpp
    src/visualizer/panel.cpp
    src/visualizer/graph.cpp
)

target_include_directories(NeuralNetwork
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_definitions(NeuralNetwork
    PRIVATE "RESOURCE_DIR=\"${NN_RESOURCE_DIR_PATH}\""
)

target_link_libraries(NeuralNetwork
    PUBLIC
        SFML::Graphics
        SFML::Window
        SFML::System
        nlohmann_json::nlohmann_json
)

target_compile_options(NeuralNetwork PRIVATE -Wall -Wextra -Wpedantic)

# ------------------------------------------------------------------
# Tests
enable_testing()
include(CTest)

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS tests/*.cpp)

if(TEST_SOURCES)
    foreach(test_file ${TEST_SOURCES})
        get_filename_component(test_name ${test_file} NAME_WE)

        add_executable(${test_name} ${test_file})

        target_link_libraries(${test_name} PRIVATE NeuralNetwork)

        target_include_directories(${test_name} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )

        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
else()
    message(WARNING "No test sources found in tests/ directory.")
endif()

# ------------------------------------------------------------------
# Install
install(TARGETS NeuralNetwork ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

