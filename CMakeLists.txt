cmake_minimum_required(VERSION 3.10)
project(NeuralNetwork LANGUAGES CXX)

# --- Settings ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(NN_RESOURCE_DIR_PATH "${CMAKE_CURRENT_SOURCE_DIR}/resources/fonts")

find_package(SFML 2.5 COMPONENTS system window graphics REQUIRED)

add_library(NeuralNetwork STATIC
    src/model/AiModel.cpp
    src/model/model.cpp
    src/trainer/trainer.cpp
    src/trainer/dataBase.cpp
    src/model/activations.cpp
    src/trainer/backPropagation.cpp
    src/model/neuralNetwork.cpp
    src/trainer/gradient.cpp
    src/model/LayerParameters.cpp
    src/model/neuron.cpp
    src/model/Layers/Hidden_Layer.cpp
    src/model/Layers/layer.cpp
    src/model/Layers/Output_Layer.cpp
    src/visualizer/VisualizerRenderer.cpp
    src/visualizer/VisualizerController.cpp
    src/visualizer/visualNN.cpp
    src/visualizer/visualL.cpp
    src/visualizer/state.cpp
    src/visualizer/VInterface.cpp
    src/visualizer/button.cpp
    src/visualizer/fonts.cpp
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

if(MSVC)
    target_compile_options(NeuralNetwork PRIVATE /W4)
else()
    target_compile_options(NeuralNetwork PRIVATE -Wall -Wextra -Wpedantic)
endif()


if(BUILD_TESTING)
    enable_testing()
    include(CTest)

    file(GLOB TEST_SOURCES
        RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
        "tests/*.cpp"
    )

    if(NOT TEST_SOURCES)
        message(WARNING "No test sources found in tests/ directory.")
    else()
        add_executable(run_tests ${TEST_SOURCES})

        # Link your NeuralNetwork library AND the SFML libraries to run_tests
        target_link_libraries(run_tests PRIVATE
            NeuralNetwork
            SFML::system
            SFML::window
            SFML::graphics
            # Add SFML::network and SFML::audio if your tests or visualizer use them
        )

        # Ensure tests can find your library's public headers and SFML headers
        target_include_directories(run_tests PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${SFML_INCLUDE_DIR} # Good practice if tests directly use SFML
        )
        add_test(NAME all_tests COMMAND run_tests)
    endif()
endif()

install(TARGETS NeuralNetwork ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
