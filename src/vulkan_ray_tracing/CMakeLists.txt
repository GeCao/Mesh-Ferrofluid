cmake_minimum_required(VERSION 3.10)

project(VulkanRayTracingApp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_VERBOSE_MAKEFILE OFF)

# Vulkan SDK
# find_package(Vulkan REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(OpenMP REQUIRED)

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
add_executable(VulkanRayTracingApp main.cpp)
# Create the Python module
pybind11_add_module(VulkanRayTracing pybind.cpp)

# Then link its targets
target_link_libraries(VulkanRayTracingApp
    PRIVATE
    vulkan
    ${Vulkan_LIBRARIES}
    OpenMP::OpenMP_CXX
)
target_link_libraries(VulkanRayTracing
    PRIVATE
    vulkan
    ${Vulkan_LIBRARIES}
    OpenMP::OpenMP_CXX
    Python3::Python
)

# Set the output directory for shaders
set(SHADER_DIR "${CMAKE_SOURCE_DIR}/spv")
set(GLSLANG_VALIDATOR "${CMAKE_SOURCE_DIR}/build/external/glslang/build/StandAlone/glslangValidator")
# Verify it exists
if(NOT EXISTS ${GLSLANG_VALIDATOR})
  message(FATAL_ERROR "glslangValidator not found at ${GLSLANG_VALIDATOR}")
endif()

# Add custom command to compile shaders
add_custom_command(
    TARGET VulkanRayTracingApp PRE_BUILD
    COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.3 ${CMAKE_SOURCE_DIR}/shaders/raygen.rgen      -o ${SHADER_DIR}/raygen.spv
    COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.3 ${CMAKE_SOURCE_DIR}/shaders/miss.rmiss       -o ${SHADER_DIR}/miss.spv
    COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.3 ${CMAKE_SOURCE_DIR}/shaders/closesthit.rchit -o ${SHADER_DIR}/closesthit.spv
    COMMENT "Compiling shaders..."
)
install(TARGETS VulkanRayTracing LIBRARY DESTINATION ${CMAKE_SOURCE_DIR})
