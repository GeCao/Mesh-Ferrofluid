cmake_minimum_required(VERSION 3.10)

project(VulkanRayQueryApp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_VERBOSE_MAKEFILE OFF)

# Vulkan SDK
# find_package(Vulkan REQUIRED)
find_program(Python3_EXECUTABLE NAMES python3 python)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import sys; print(sys.executable)"
    OUTPUT_VARIABLE PYTHON_PATH OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(Python3_EXECUTABLE ${PYTHON_PATH} CACHE FILEPATH "Path to python executable")

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)


# Include directories for Vulkan and shaders
include_directories(
    ${Vulkan_INCLUDE_DIRS}
    external/glm
)

# Set the libraries for Vulkan and shaders
link_directories(${Vulkan_LIBRARIES})

# Add glslang project directly from external
add_subdirectory(external/pybind11)
add_subdirectory(external/glslang EXCLUDE_FROM_ALL)

# Add source file
add_executable(VulkanRayQueryApp main.cpp)
# Create the Python module
pybind11_add_module(VulkanRayQuery pybind.cpp)

# Then link its targets
target_link_libraries(VulkanRayQueryApp
    PRIVATE
    vulkan
    ${Vulkan_LIBRARIES}
)
target_link_libraries(VulkanRayQuery
    PRIVATE
    vulkan
    ${Vulkan_LIBRARIES}
    Python3::Python
)

# Set the output directory for shaders
set(SHADER_DIR "${CMAKE_SOURCE_DIR}/spv")
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
set(GLSLANG_VALIDATOR "${CMAKE_SOURCE_DIR}/build/external/glslang/build/StandAlone/Release/glslangValidator.exe")
else()
    set(GLSLANG_VALIDATOR "${CMAKE_SOURCE_DIR}/build/external/glslang/build/StandAlone/glslangValidator")
endif()
# Verify it exists
if(NOT EXISTS ${GLSLANG_VALIDATOR})
  message(FATAL_ERROR "glslangValidator not found at ${GLSLANG_VALIDATOR}")
endif()

# Add custom command to compile shaders
add_custom_command(
    TARGET VulkanRayQueryApp PRE_BUILD
    COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.3 ${CMAKE_SOURCE_DIR}/shaders/raygen.rgen      -o ${SHADER_DIR}/raygen.spv
    COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.3 ${CMAKE_SOURCE_DIR}/shaders/miss.rmiss       -o ${SHADER_DIR}/miss.spv
    COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.3 ${CMAKE_SOURCE_DIR}/shaders/closesthit.rchit -o ${SHADER_DIR}/closesthit.spv
    COMMENT "Compiling shaders..."
)
install(TARGETS VulkanRayQuery LIBRARY DESTINATION ${CMAKE_SOURCE_DIR})
