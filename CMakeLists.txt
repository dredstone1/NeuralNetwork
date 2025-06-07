cmake_minimum_required(VERSION 3.28)
project(NeuralNetwork LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(NN_RESOURCE_DIR_PATH "${CMAKE_CURRENT_SOURCE_DIR}/resources/fonts")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(SANITIZER_FLAGS "-fsanitize=address -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SANITIZER_FLAGS}")
set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} ${SANITIZER_FLAGS}")

include(FetchContent)
FetchContent_Declare(SFML
    GIT_REPOSITORY https://github.com/SFML/SFML.git
    GIT_TAG 3.0.1
    GIT_SHALLOW ON
    EXCLUDE_FROM_ALL
    SYSTEM
)
FetchContent_MakeAvailable(SFML)

find_package(nlohmann_json CONFIG REQUIRED)

add_library(NeuralNetwork STATIC
    src/model/AiModel.cpp
    src/model/model.cpp
    src/model/activations.cpp
    src/model/neuralNetwork.cpp
    src/model/LayerParameters.cpp
    src/model/neuron.cpp
    src/model/Layers/Hidden_Layer.cpp
    src/model/Layers/layer.cpp
    src/model/Layers/Output_Layer.cpp
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
        ${SFML_INCLUDE_DIR}
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

add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
add_link_options(-fsanitize=address)

enable_testing()
include(CTest)

file(GLOB TEST_SOURCES
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "tests/*.cpp"
)

if(NOT TEST_SOURCES)
    message(WARNING "No test sources found in tests/ directory.")
else()
    foreach(test_file ${TEST_SOURCES})
        get_filename_component(test_name ${test_file} NAME_WE)

        add_executable(${test_name} ${test_file})

        target_link_libraries(${test_name} PRIVATE
            NeuralNetwork
            sfml-graphics
            sfml-window
            sfml-system
            nlohmann_json::nlohmann_json
        )

        target_include_directories(${test_name} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${SFML_INCLUDE_DIR}
        )

        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()

install(TARGETS NeuralNetwork ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
