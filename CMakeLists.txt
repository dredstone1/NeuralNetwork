cmake_minimum_required(VERSION 3.28)
project(NeuralNetwork LANGUAGES CXX)

# ------------------------------------------------------------------
# Configuration
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Default to Debug build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# ------------------------------------------------------------------
# Paths
set(NN_RESOURCE_DIR_PATH "${CMAKE_CURRENT_SOURCE_DIR}/resources/fonts")

# ------------------------------------------------------------------
# Dependencies (FetchContent)
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
file(GLOB_RECURSE NN_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

add_library(NeuralNetwork STATIC ${NN_SOURCES})
set_target_properties(NeuralNetwork PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(NeuralNetwork
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
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
# Sanitizers (enabled in Debug mode only)
option(ENABLE_SANITIZERS "Enable address sanitizer" ON)
if(ENABLE_SANITIZERS AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(NeuralNetwork PRIVATE -fsanitize=address -fno-omit-frame-pointer -g)
    target_link_options(NeuralNetwork PRIVATE -fsanitize=address)
endif()

# ------------------------------------------------------------------
# Tests
option(BUILD_NN_TESTS "Build NeuralNetwork tests" OFF)

if(BUILD_NN_TESTS)
    enable_testing()
    include(CTest)

    file(GLOB TEST_SOURCES CONFIGURE_DEPENDS tests/*.cpp)

    if(TEST_SOURCES)
        foreach(test_file ${TEST_SOURCES})
            get_filename_component(test_name "${test_file}" NAME_WE)

            add_executable(${test_name} ${test_file})

            target_link_libraries(${test_name} PRIVATE NeuralNetwork)

            target_include_directories(${test_name} PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/include"
            )

            add_test(NAME ${test_name} COMMAND ${test_name})
        endforeach()
    else()
        message(WARNING "No test sources found in tests/ directory.")
    endif()
endif()

# ------------------------------------------------------------------
# Install
install(TARGETS NeuralNetwork
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)

