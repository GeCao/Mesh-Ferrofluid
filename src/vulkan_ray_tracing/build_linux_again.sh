#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Create and Navigate to build directory if it doesn't exist
BUILD_DIR=build
cd $BUILD_DIR

# External libraries:
# 1.1 The libraries coming from apt install:
# 1.1.1 Set Vulkan SDK path
VULKAN_SDK_INCLUDE=$VULKAN_SDK/include
VULKAN_SDK_LIB=$VULKAN_SDK/lib

echo $VULKAN_SDK
echo $VULKAN_SDK_INCLUDE
echo $VULKAN_SDK_LIB

# 1.2 The libraries coming from submodules:
EXT_PATH=external
# rm -rf ./$EXT_PATH && mkdir -p $EXT_PATH

# 1.2.1 Set SPIRV path (adjust according to your system)
SPIRV_HEADERs_BIN=$EXT_PATH/spirv-headers
# cp -r ../$SPIRV_HEADERs_BIN $SPIRV_HEADERs_BIN
# mkdir -p $SPIRV_HEADERs_BIN/build
# cd $SPIRV_HEADERs_BIN/build
# cmake ..
# make -j$(nproc)
# cd ../../../

SPIRV_BIN=$EXT_PATH/spirv-tools
# cp -r ../$SPIRV_BIN $SPIRV_BIN
# mkdir -p $SPIRV_BIN/build
# cd $SPIRV_BIN/build
# cmake -DSPIRV-Headers_SOURCE_DIR=${PWD}/../../spirv-headers ..
# make -j$(nproc)
# cd ../../../

# 1.2.2 Set GLSLang path
GLSLANG_BIN=$EXT_PATH/glslang
# cp -r ../$GLSLANG_BIN $GLSLANG_BIN
# mkdir -p $GLSLANG_BIN/build
# cd $GLSLANG_BIN/build
# cmake \
#   -DENABLE_OPT=OFF \
#   -DSPIRV_TOOLS_INCLUDE_DIR=${PWD}/../../spirv-tools/include \
#   -DSPIRV_TOOLS_LIB_DIR=${PWD}/../../spirv-tools/build \
#   ..
# make -j$(nproc)
# cd ../../../

# 1.2.3 Set tbb path by vcpkg
VCPKG_BIN=$EXT_PATH/vcpkg
# cp -r ../$VCPKG_BIN $VCPKG_BIN
# cd $VCPKG_BIN
# ./bootstrap-vcpkg.sh #.\bootstrap-vcpkg.bat(for Windows)
# ./vcpkg integrate install
# ./vcpkg install tbb
# cd ../../

# 1.2.5 Set glm path
GLM_BIN=$EXT_PATH/glm
cp -r ../$GLM_BIN $GLM_BIN

# Set environment variables for Vulkan and GLSLang
export VK_ICD_FILENAMES=$VULKAN_SDK/etc/vulkan/icd.d/nvidia_icd.json
export VK_LAYER_PATH=$VULKAN_SDK/etc/vulkan/explicit_layer.d

# Run cmake to configure the project
echo "Building main Vulkan project..."
cmake \
  -DENABLE_OPT=OFF \
  -DGLM_INCLUDE_DIR=${PWD}/$EXT_PATH/glm \
  -DCMAKE_PREFIX_PATH=${PWD}/$EXT_PATH/vcpkg/installed/x64-linux \
  -DBASE_DIRECTORY=${PWD}/$EXT_PATH -DCMAKE_BUILD_TYPE=Release \
  -DVulkan_INCLUDE_DIRS=$VULKAN_SDK/include -DVulkan_LIBRARIES=$VULKAN_SDK_LIB \
  -DPython3_EXECUTABLE=$(which python) \
  ..

# Compile the project using make
make install

# Optionally, you can clean up and prepare for execution
echo "Build complete. You can now run your Vulkan app."
