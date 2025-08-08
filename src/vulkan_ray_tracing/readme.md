# Vulkan-based ray-querier


#### Install vulkan for your linux system at first:
1. Go find your vulkan version from: https://vulkan.lunarg.com/sdk/home and install.

    or just use command "sudo apt install vulkan-tools libvulkan-dev mesa-vulkan-drivers" if you are using Ubuntu.

    However, the first choice would be our recommendation as it is the newest.
    
    Remember to add following to your environment
    ```bash
    source PATH_TO_VULKAN_SDK/setup-env.sh
    export VULKAN_SDK=/x86_64
    ```

2. Install the necessary libraries:
```bash
sudo apt-get install curl zip unzip tar
sudo apt install libglfw3-dev nlohmann-json3-dev libglm-dev libfmt-dev libxinerama-dev libxcursor-dev libxi-dev
#  sudo apt install libtbb-dev glslang-dev glslang-tools 
git submodule update --init --recursive

# Install the newest version of tbb
cd external/vcpkg
./bootstrap-vcpkg.sh #.\bootstrap-vcpkg.bat(for Windows)
./vcpkg integrate install
./vcpkg install tbb
cd ../../
```

### Compile the main function as the first attempt.
```
bash build_linux.sh
build/VulkanRayQueryApp
```

### Trouble shooting: