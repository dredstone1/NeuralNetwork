cmake_minimum_required(VERSION 3.10)
project(NeuralNetwork LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(NN_RESOURCE_DIR_PATH "${CMAKE_CURRENT_SOURCE_DIR}/resources/fonts")

find_package(SFML 2.6 COMPONENTS system window graphics REQUIRED)

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


# --- Test configuration - Always create test executable ---
enable_testing()
include(CTest)

file(GLOB TEST_SOURCES
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "tests/*.cpp"
)

if(NOT TEST_SOURCES)
    message(WARNING "No test sources found in tests/ directory. Creating a dummy test executable that returns 0.")
    # Define content for a dummy main.cpp
    set(DUMMY_MAIN_CONTENT "int main() {\n  // No actual tests found, returning success.\n  return 0;\n}\n")
    # Define path for the dummy main.cpp in the build directory
    set(DUMMY_MAIN_FILE_PATH "${CMAKE_CURRENT_BINARY_DIR}/dummy_test_main.cpp")
    # Write the content to the dummy file
    file(WRITE "${DUMMY_MAIN_FILE_PATH}" "${DUMMY_MAIN_CONTENT}")
    # Add the executable with the dummy main file
    add_executable(run_tests "${DUMMY_MAIN_FILE_PATH}")
else()
    # Add the executable with the actual test sources
    add_executable(run_tests ${TEST_SOURCES})
endif()

# Common setup for the run_tests executable, which is now guaranteed to be defined
target_link_libraries(run_tests PRIVATE
    NeuralNetwork
    sfml-graphics
    sfml-window
    sfml-system
)
target_include_directories(run_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${SFML_INCLUDE_DIR}
)

# Add the test to CTest. It will run the (potentially dummy) executable.
add_test(NAME all_tests COMMAND run_tests)
# --- End of Test configuration ---


install(TARGETS NeuralNetwork ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
