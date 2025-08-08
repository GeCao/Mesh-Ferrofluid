#ifndef VULKAN_RAY_TRACING_RAYQUERYAPP
#define VULKAN_RAY_TRACING_RAYQUERYAPP

#include <vulkan/vulkan.h>
#include <iostream>
#include <vector>
#include <array>
#include <cassert>
#include <cstring>
#include <glm/glm.hpp>

#include "helper.hpp"

const char *deviceExtensions[] = {
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
    VK_KHR_SPIRV_1_4_EXTENSION_NAME,
    VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME,
};

bool enableValidationLayers = true;
const std::vector<const char *> validationLayers = {"VK_LAYER_KHRONOS_validation"};

class RayQueryApp
{
public:
    // Instance
    VkInstance instance;

    // Devices
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;

    // queue
    VkQueue computeQueue;
    uint32_t queueFamilyIndex;

    // Commands
    VkCommandPool commandPool;
    VkFence fence;

    // Geometries
    std::vector<Vertex> txs;
    std::vector<Vertex> rayDirs;

    // BLAS and TLAS
    AccelStructure blas;
    AccelStructure tlas;

    // Buffers
    Buffer vertexBuffer, indicesBuffer, alignedIndicesBuffer, txBuffer, rayDirsBuffer;                                    // Geometry related, create when building BLAS
    Buffer vertexStagingBuffer, indicesStagingBuffer, alignedIndicesStagingBuffer, txStagingBuffer, rayDirsStagingBuffer; // Geometry related, create when building BLAS
    Buffer occlusionBuffer, occlusionStagingBuffer, scratchBlasBuffer;                                                    // Output related, create when building BLAS
    Buffer instanceBuffer, stagingTlasBuffer, scratchTlasBuffer;                                                          // Create when building TLAS
    Buffer sbtBuffer;                                                                                                     // Create when building rayPipeline
    ObjDesc objDescData = {0, 0, 0, 0};

    // Shaders
    VkShaderModule raygenModule;
    VkShaderModule missModule;
    VkShaderModule chitModule;
    VkStridedDeviceAddressRegionKHR raygenRegion;
    VkStridedDeviceAddressRegionKHR missRegion;
    VkStridedDeviceAddressRegionKHR hitRegion;
    VkStridedDeviceAddressRegionKHR callableRegion{0, 0, 0};

    // Pipelines
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;
    VkPipelineLayout pipelineLayout;
    VkPipeline rayPipeline;

    RayQueryApp(
        const std::vector<float> &vertices,
        const std::vector<uint32_t> &indices,
        const std::string &raygenShaderPath,
        const std::string &missShaderPath,
        const std::string &chitShaderPath)
    {
        createInstance();
        createPhysicalAndLogicDevice(); // As well as queue, included
        createCommandPool();
        createFence();

        // Create BLAS and TLAS
        blas = createBLAS(vertices, indices);
        tlas = createTLAS();

        // // Create TLAS and BLAS:
        // VkCommandBuffer cmdBuf_build = createNewCommandBuffer();
        // beginCommand(cmdBuf_build);

        // vkEndCommandBuffer(cmdBuf_build);
        // // Submit command buffer
        // VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        // submitInfo.commandBufferCount = 1;
        // submitInfo.pCommandBuffers = &cmdBuf_build;
        // vkQueueSubmit(computeQueue, 1, &submitInfo, fence);
        // vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

        // Clean all the used staging buffers
        // vkFreeCommandBuffers(device, commandPool, 1, &cmdBuf_build);
        vkDestroyBuffer(device, vertexStagingBuffer.buffer, nullptr);
        vkFreeMemory(device, vertexStagingBuffer.memory, nullptr);
        vkDestroyBuffer(device, indicesStagingBuffer.buffer, nullptr);
        vkFreeMemory(device, indicesStagingBuffer.memory, nullptr);
        vkDestroyBuffer(device, alignedIndicesStagingBuffer.buffer, nullptr);
        vkFreeMemory(device, alignedIndicesStagingBuffer.memory, nullptr);
        vkDestroyBuffer(device, stagingTlasBuffer.buffer, nullptr);
        vkFreeMemory(device, stagingTlasBuffer.memory, nullptr);

        descriptorSetLayout = createDescriptorSetLayout(device);
        createDescriptorPool();
        descriptorSet = createDescriptor();
        pipelineLayout = createPipelineLayout(device, descriptorSetLayout);
        rayPipeline = createRayPipeline(
            device, pipelineLayout, raygenModule, missModule, chitModule, raygenShaderPath, missShaderPath, chitShaderPath);
        buildSBT(); // Only relies on rayPipeline
    }

    void createInstance()
    {
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "HeadlessRayQuery";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_3;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        if (enableValidationLayers)
        {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else
        {
            createInfo.enabledLayerCount = 0;
        }

        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create instance!");
        }
    }

    void createCommandPool()
    {
        // Create Command pool
        VkCommandPoolCreateInfo poolCreateInfo = {};
        poolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolCreateInfo.queueFamilyIndex = queueFamilyIndex;                     // The index of the queue family for graphics commands.
        poolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; // Allows resetting the command buffers.

        if (vkCreateCommandPool(device, &poolCreateInfo, nullptr, &commandPool) != VK_SUCCESS)
        {
            std::cerr << "Failed to create command pool!" << std::endl;
            return;
        }
    }

    void createFence()
    {
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

        vkCreateFence(device, &fenceInfo, nullptr, &fence);
    }

    void createPhysicalAndLogicDevice()
    {
        // Step 1.0: Check how many devices you got here:
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        if (deviceCount == 0)
        {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        // Step 1.2: If any device detected, declare it as PhysicalDevice!
        VkPhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddressFeatures{};
        bufferDeviceAddressFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
        bufferDeviceAddressFeatures.bufferDeviceAddress = VK_TRUE;

        VkPhysicalDeviceAccelerationStructureFeaturesKHR accelStructFeatures{};
        accelStructFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
        accelStructFeatures.accelerationStructure = VK_TRUE;
        accelStructFeatures.pNext = &bufferDeviceAddressFeatures;

        VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipelineFeatures{};
        rayTracingPipelineFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
        rayTracingPipelineFeatures.rayTracingPipeline = VK_TRUE;
        rayTracingPipelineFeatures.pNext = &accelStructFeatures; // Chain them

        VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures{};
        rayQueryFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
        rayQueryFeatures.rayQuery = VK_TRUE;
        rayQueryFeatures.pNext = &rayTracingPipelineFeatures;
        // rayQueryFeatures.pNext = &accelStructFeatures;

        for (const auto &dev : devices)
        {
            VkPhysicalDeviceFeatures2 features2{};
            features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            features2.pNext = &rayQueryFeatures;
            features2.features.shaderInt64 = VK_TRUE;
            vkGetPhysicalDeviceFeatures2(dev, &features2);

            if (rayQueryFeatures.rayQuery == VK_TRUE)
            {
                physicalDevice = dev;
                break;
            }
        }

        // Step 1.3: No device found, throw an error
        if (physicalDevice == VK_NULL_HANDLE)
        {
            throw std::runtime_error("failed to find a suitable GPU with rayQuery support");
        }

        // Step 2.0: Queue
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());
        std::cout << "queueFamilyCount = " << queueFamilyCount << std::endl; // = 4
        queueFamilyIndex = UINT32_MAX;
        for (uint32_t i = 0; i < queueFamilyCount; i++)
        {
            const auto &qf = queueFamilies[i];
            // Make sure it's compute-capable and not graphics-capable
            if ((qf.queueFlags & VK_QUEUE_COMPUTE_BIT) &&
                !(qf.queueFlags & VK_QUEUE_GRAPHICS_BIT))
            {
                queueFamilyIndex = i; // = 2, VK_QUEUE_COMPUTE_BIT
                break;
            }
        }
        std::cout << "queueFamilyIndex = " << queueFamilyIndex << std::endl;
        float queuePriority = 1.f;
        VkDeviceQueueCreateInfo queueCreateInfos[1] = {};
        queueCreateInfos[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfos[0].queueCount = 1;
        queueCreateInfos[0].pQueuePriorities = &queuePriority;
        queueCreateInfos[0].queueFamilyIndex = queueFamilyIndex;

        // Step 3.0: Create logical device
        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.pQueueCreateInfos = queueCreateInfos;
        createInfo.queueCreateInfoCount = 1;
        createInfo.pNext = &rayQueryFeatures; // TODO: &physicalDeviceFeatures
        createInfo.ppEnabledExtensionNames = deviceExtensions;
        createInfo.enabledExtensionCount = sizeof(deviceExtensions) / sizeof(deviceExtensions[0]);

        VkPhysicalDeviceFeatures deviceFeatures{};
        deviceFeatures.shaderInt64 = VK_TRUE;
        createInfo.pEnabledFeatures = &deviceFeatures;

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create logical device!");
        }

        vkGetDeviceQueue(device, queueFamilyIndex, 0, &computeQueue);
    }

    VkCommandBuffer createNewCommandBuffer()
    {
        VkCommandBuffer cmdBuf; // This is where the command buffer will be stored.

        VkCommandBufferAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; // Primary command buffer (the one you can submit).
        allocInfo.commandBufferCount = 1;                  // Number of command buffers to allocate.

        if (vkAllocateCommandBuffers(device, &allocInfo, &cmdBuf) != VK_SUCCESS)
        {
            std::cerr << "Failed to allocate command buffer!" << std::endl;
            return cmdBuf;
        }

        return cmdBuf;
    }

    void createTxRaydirOcclusionBuffers(VkCommandBuffer cmdBuf, std::vector<Vertex> &txs, std::vector<Vertex> &rayDirs)
    {
        uint32_t height = txs.size();
        uint32_t width = rayDirs.size() / height;

        txBuffer.size = sizeof(Vertex) * txs.size();
        rayDirsBuffer.size = sizeof(Vertex) * rayDirs.size();
        occlusionBuffer.size = sizeof(float) * 4 * height * width; // 30 Tx × 20 rayDirs
        txStagingBuffer.size = txBuffer.size;
        rayDirsStagingBuffer.size = rayDirsBuffer.size;
        occlusionStagingBuffer.size = occlusionBuffer.size;

        createBuffer(device, physicalDevice, txBuffer.size,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, txBuffer.buffer, txBuffer.memory);
        createBuffer(device, physicalDevice, rayDirsBuffer.size,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, rayDirsBuffer.buffer, rayDirsBuffer.memory);
        createBuffer(device, physicalDevice, occlusionBuffer.size,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                         VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     occlusionBuffer.buffer, occlusionBuffer.memory);

        createBuffer(device, physicalDevice, txStagingBuffer.size,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, txStagingBuffer.buffer, txStagingBuffer.memory);
        createBuffer(device, physicalDevice, rayDirsStagingBuffer.size,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, rayDirsStagingBuffer.buffer, rayDirsStagingBuffer.memory);
        createBuffer(device, physicalDevice, occlusionStagingBuffer.size,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                         VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     occlusionStagingBuffer.buffer, occlusionStagingBuffer.memory);

        txBuffer.deviceAddress = getBufferDeviceAddress(device, txBuffer.buffer);
        rayDirsBuffer.deviceAddress = getBufferDeviceAddress(device, rayDirsBuffer.buffer);
        occlusionBuffer.deviceAddress = getBufferDeviceAddress(device, occlusionBuffer.buffer);

        objDescData.txAddress = txBuffer.deviceAddress;
        objDescData.rayDirAddress = rayDirsBuffer.deviceAddress;

        void *tx_mapped;
        vkMapMemory(device, txStagingBuffer.memory, 0, txStagingBuffer.size, 0, &tx_mapped);
        memcpy(tx_mapped, txs.data(), txStagingBuffer.size);
        vkUnmapMemory(device, txStagingBuffer.memory);

        VkBufferCopy tx_copyRegion{};
        tx_copyRegion.size = txBuffer.size;
        vkCmdCopyBuffer(cmdBuf, txStagingBuffer.buffer, txBuffer.buffer, 1, &tx_copyRegion);

        void *rayDirs_mapped;
        vkMapMemory(device, rayDirsStagingBuffer.memory, 0, rayDirsStagingBuffer.size, 0, &rayDirs_mapped);
        memcpy(rayDirs_mapped, rayDirs.data(), rayDirsStagingBuffer.size);
        vkUnmapMemory(device, rayDirsStagingBuffer.memory);

        VkBufferCopy rayDirs_copyRegion{};
        rayDirs_copyRegion.size = rayDirsBuffer.size;
        vkCmdCopyBuffer(cmdBuf, rayDirsStagingBuffer.buffer, rayDirsBuffer.buffer, 1, &rayDirs_copyRegion);
    }

    AccelStructure createBLAS(const std::vector<float> &vertices, const std::vector<uint32_t> &indices)
    {
        VkCommandBuffer cmdBuf = createNewCommandBuffer();
        vkResetFences(device, 1, &fence);
        beginCommand(cmdBuf);

        uint32_t alignedIndicesSize = indices.size() / 3;
        std::vector<Indice> alignedIndices(alignedIndicesSize);
        for (int i = 0; i < alignedIndicesSize; ++i)
        {
            alignedIndices[i] = Indice(indices[3 * i], indices[3 * i + 1], indices[3 * i + 2]);
        }

        std::vector<Vertex> vertex_vertices(static_cast<uint32_t>(vertices.size() / 3));
        for (int i = 0; i < vertex_vertices.size(); ++i)
        {
            vertex_vertices[i] = Vertex(vertices[3 * i], vertices[3 * i + 1], vertices[3 * i + 2]);
        }

        vertexBuffer.size = sizeof(Vertex) * vertex_vertices.size();
        indicesBuffer.size = sizeof(uint32_t) * indices.size();
        alignedIndicesBuffer.size = sizeof(Indice) * alignedIndices.size();
        vertexStagingBuffer.size = vertexBuffer.size;
        indicesStagingBuffer.size = indicesBuffer.size;
        alignedIndicesStagingBuffer.size = alignedIndicesBuffer.size;
        std::cout << "alignedIndicesBuffer.size = " << alignedIndicesBuffer.size << std::endl;

        createBuffer(device, physicalDevice, vertexBuffer.size,
                     VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer.buffer, vertexBuffer.memory);
        createBuffer(device, physicalDevice, indicesBuffer.size,
                     VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indicesBuffer.buffer, indicesBuffer.memory);
        createBuffer(device, physicalDevice, alignedIndicesBuffer.size,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, alignedIndicesBuffer.buffer, alignedIndicesBuffer.memory);

        createBuffer(device, physicalDevice, vertexStagingBuffer.size,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, vertexStagingBuffer.buffer, vertexStagingBuffer.memory);
        createBuffer(device, physicalDevice, indicesStagingBuffer.size,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, indicesStagingBuffer.buffer, indicesStagingBuffer.memory);
        createBuffer(device, physicalDevice, alignedIndicesStagingBuffer.size,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, alignedIndicesStagingBuffer.buffer, alignedIndicesStagingBuffer.memory);

        // Normally you'd copy the vertex/index data to the buffers here using staging

        vertexBuffer.deviceAddress = getBufferDeviceAddress(device, vertexBuffer.buffer);
        indicesBuffer.deviceAddress = getBufferDeviceAddress(device, indicesBuffer.buffer);
        alignedIndicesBuffer.deviceAddress = getBufferDeviceAddress(device, alignedIndicesBuffer.buffer);

        objDescData.vertexAddress = vertexBuffer.deviceAddress;
        objDescData.indexAddress = alignedIndicesBuffer.deviceAddress;

        void *vertex_mapped;
        vkMapMemory(device, vertexStagingBuffer.memory, 0, vertexStagingBuffer.size, 0, &vertex_mapped);
        memcpy(vertex_mapped, vertex_vertices.data(), vertexStagingBuffer.size);
        vkUnmapMemory(device, vertexStagingBuffer.memory);

        VkBufferCopy vertex_copyRegion{};
        vertex_copyRegion.size = vertexBuffer.size;
        vkCmdCopyBuffer(cmdBuf, vertexStagingBuffer.buffer, vertexBuffer.buffer, 1, &vertex_copyRegion);

        void *index_mapped;
        vkMapMemory(device, indicesStagingBuffer.memory, 0, indicesStagingBuffer.size, 0, &index_mapped);
        memcpy(index_mapped, indices.data(), indicesStagingBuffer.size);
        vkUnmapMemory(device, indicesStagingBuffer.memory);

        VkBufferCopy indices_copyRegion{};
        indices_copyRegion.size = indicesBuffer.size;
        vkCmdCopyBuffer(cmdBuf, indicesStagingBuffer.buffer, indicesBuffer.buffer, 1, &indices_copyRegion);

        void *aligned_index_mapped;
        vkMapMemory(device, alignedIndicesStagingBuffer.memory, 0, alignedIndicesStagingBuffer.size, 0, &aligned_index_mapped);
        memcpy(aligned_index_mapped, alignedIndices.data(), alignedIndicesStagingBuffer.size);
        vkUnmapMemory(device, alignedIndicesStagingBuffer.memory);

        VkBufferCopy alignedIndices_copyRegion{};
        alignedIndices_copyRegion.size = alignedIndicesBuffer.size;
        vkCmdCopyBuffer(cmdBuf, alignedIndicesStagingBuffer.buffer, alignedIndicesBuffer.buffer, 1, &alignedIndices_copyRegion);

        VkAccelerationStructureGeometryTrianglesDataKHR triangles{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
        triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        triangles.vertexData.deviceAddress = vertexBuffer.deviceAddress;
        triangles.vertexStride = sizeof(Vertex);
        triangles.indexType = VK_INDEX_TYPE_UINT32;
        triangles.indexData.deviceAddress = indicesBuffer.deviceAddress;
        triangles.maxVertex = static_cast<uint32_t>(vertices.size() - 1);
        std::cout << "triangles.maxVertex = " << triangles.maxVertex << std::endl;

        VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        geometry.geometry.triangles = triangles;

        VkAccelerationStructureBuildGeometryInfoKHR buildInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        buildInfo.srcAccelerationStructure = VK_NULL_HANDLE; // No need in BLAS
        buildInfo.dstAccelerationStructure = VK_NULL_HANDLE; // Required in BLAS, TBD in following code
        buildInfo.geometryCount = 1;
        buildInfo.pGeometries = &geometry;
        buildInfo.scratchData.deviceAddress = 0; // Required in BLAS, TBD in following code

        uint32_t primitiveCount = static_cast<uint32_t>(indices.size()) / 3; // Num of triangles

        VkAccelerationStructureBuildSizesInfoKHR sizeInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        PFN_vkGetAccelerationStructureBuildSizesKHR vkGetAccelerationStructureBuildSizesKHR =
            (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR");
        vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                                &buildInfo, &primitiveCount, &sizeInfo);

        AccelStructure accelStruct;
        createBuffer(device, physicalDevice, sizeInfo.accelerationStructureSize,
                     VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, accelStruct.buffer, accelStruct.memory);

        VkAccelerationStructureCreateInfoKHR createInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
        createInfo.buffer = accelStruct.buffer;
        createInfo.size = sizeInfo.accelerationStructureSize;
        createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR =
            (PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR");
        vkCreateAccelerationStructureKHR(device, &createInfo, nullptr, &accelStruct.handle);

        buildInfo.dstAccelerationStructure = accelStruct.handle;

        VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
        rangeInfo.firstVertex = 0;
        rangeInfo.primitiveCount = primitiveCount;
        rangeInfo.primitiveOffset = 0;
        rangeInfo.transformOffset = 0;
        const VkAccelerationStructureBuildRangeInfoKHR *rangeInfos[] = {&rangeInfo};

        scratchBlasBuffer.size = sizeInfo.buildScratchSize;
        createBuffer(device, physicalDevice, scratchBlasBuffer.size,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, scratchBlasBuffer.buffer, scratchBlasBuffer.memory);
        scratchBlasBuffer.deviceAddress = getBufferDeviceAddress(device, scratchBlasBuffer.buffer);
        buildInfo.scratchData.deviceAddress = scratchBlasBuffer.deviceAddress;
        std::cout << "[Building BLAS] sizeInfo.accelerationStructureSize = " << sizeInfo.accelerationStructureSize << ", sizeInfo.buildScratchSize = " << sizeInfo.buildScratchSize << std::endl;

        PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR =
            (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR");
        vkCmdBuildAccelerationStructuresKHR(cmdBuf, 1, &buildInfo, rangeInfos);

        vkEndCommandBuffer(cmdBuf);
        // Submit command buffer
        VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmdBuf;
        vkQueueSubmit(computeQueue, 1, &submitInfo, fence);
        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

        accelStruct.deviceAddress = getAccelerationStructureDeviceAddress(device, accelStruct.handle);

        return accelStruct;
    }

    // TLAS creation
    AccelStructure createTLAS()
    {
        VkCommandBuffer cmdBuf = createNewCommandBuffer();
        vkResetFences(device, 1, &fence);
        beginCommand(cmdBuf);

        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};

        VkTransformMatrixKHR transformMatrix = {
            {{1.0f, 0.0f, 0.0f, 0.0f},
             {0.0f, 1.0f, 0.0f, 0.0f},
             {0.0f, 0.0f, 1.0f, 0.0f}}};

        VkAccelerationStructureInstanceKHR instance{};
        instance.transform = transformMatrix;
        instance.instanceCustomIndex = 0;
        instance.mask = 0xFF;
        instance.instanceShaderBindingTableRecordOffset = 0;
        instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        instance.accelerationStructureReference = blas.deviceAddress;
        std::vector<VkAccelerationStructureInstanceKHR> instances{instance};

        instanceBuffer.size = sizeof(VkAccelerationStructureInstanceKHR) * instances.size();
        createBuffer(device, physicalDevice, instanceBuffer.size,
                     VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, instanceBuffer.buffer, instanceBuffer.memory);
        instanceBuffer.deviceAddress = getBufferDeviceAddress(device, instanceBuffer.buffer);

        stagingTlasBuffer.size = instanceBuffer.size;
        createBuffer(device, physicalDevice, stagingTlasBuffer.size,
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingTlasBuffer.buffer, stagingTlasBuffer.memory);
        // stagingTlasBuffer.deviceAddress = getBufferDeviceAddress(device, stagingTlasBuffer.buffer);

        VkAccelerationStructureGeometryInstancesDataKHR instancesData{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR};
        instancesData.arrayOfPointers = VK_FALSE;
        instancesData.data.deviceAddress = instanceBuffer.deviceAddress;

        VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        geometry.geometry.instances = instancesData;

        VkAccelerationStructureBuildGeometryInfoKHR buildInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        buildInfo.srcAccelerationStructure = VK_NULL_HANDLE; // No need in TLAS
        buildInfo.dstAccelerationStructure = VK_NULL_HANDLE; // Required in TLAS, TBD in following code
        buildInfo.geometryCount = 1;
        buildInfo.pGeometries = &geometry;
        buildInfo.scratchData.deviceAddress = 0; // Required in TLAS, TBD in following code

        uint32_t primitiveCount = static_cast<uint32_t>(instances.size());

        VkAccelerationStructureBuildSizesInfoKHR sizeInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        PFN_vkGetAccelerationStructureBuildSizesKHR vkGetAccelerationStructureBuildSizesKHR =
            (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR");
        vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                                &buildInfo, &primitiveCount, &sizeInfo);

        AccelStructure accelStruct;
        createBuffer(device, physicalDevice, sizeInfo.accelerationStructureSize,
                     VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, accelStruct.buffer, accelStruct.memory);

        VkAccelerationStructureCreateInfoKHR createInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
        createInfo.buffer = accelStruct.buffer;
        createInfo.size = sizeInfo.accelerationStructureSize;
        createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR =
            (PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR");
        vkCreateAccelerationStructureKHR(device, &createInfo, nullptr, &accelStruct.handle);

        buildInfo.dstAccelerationStructure = accelStruct.handle;

        VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
        rangeInfo.primitiveCount = primitiveCount;
        const VkAccelerationStructureBuildRangeInfoKHR *rangeInfos[] = {&rangeInfo};

        scratchTlasBuffer.size = sizeInfo.buildScratchSize;
        createBuffer(device, physicalDevice, scratchTlasBuffer.size,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, scratchTlasBuffer.buffer, scratchTlasBuffer.memory);
        scratchTlasBuffer.deviceAddress = getBufferDeviceAddress(device, scratchTlasBuffer.buffer);
        buildInfo.scratchData.deviceAddress = scratchTlasBuffer.deviceAddress;
        std::cout << "[Building TLAS] sizeInfo.accelerationStructureSize = " << sizeInfo.accelerationStructureSize << ", sizeInfo.buildScratchSize = " << sizeInfo.buildScratchSize << std::endl;

        void *mapped = nullptr;
        vkMapMemory(device, stagingTlasBuffer.memory, 0, stagingTlasBuffer.size, 0, &mapped);
        memcpy(mapped, instances.data(), stagingTlasBuffer.size);
        vkUnmapMemory(device, stagingTlasBuffer.memory);

        VkBufferCopy copyRegion{};
        copyRegion.size = instanceBuffer.size;
        vkCmdCopyBuffer(cmdBuf, stagingTlasBuffer.buffer, instanceBuffer.buffer, 1, &copyRegion);
        // endSingleTimeCommands(device, commandPool, computeQueue, cmdBuf);

        PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR =
            (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR");
        vkCmdBuildAccelerationStructuresKHR(cmdBuf, 1, &buildInfo, rangeInfos);

        vkEndCommandBuffer(cmdBuf);
        // Submit command buffer
        VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmdBuf;
        vkQueueSubmit(computeQueue, 1, &submitInfo, fence);
        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

        accelStruct.deviceAddress = getAccelerationStructureDeviceAddress(device, accelStruct.handle);

        return accelStruct;
    }

    void createDescriptorPool()
    {
        VkDescriptorPoolSize poolSize[] = {
            {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1}, // TLAS
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}              // Occlusion buffer
        };

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 2;
        poolInfo.pPoolSizes = poolSize;
        poolInfo.maxSets = 1;

        vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool);
    }

    VkDescriptorSet createDescriptor()
    {
        VkDescriptorSet newDescriptorSet;

        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &descriptorSetLayout;

        VkResult result = vkAllocateDescriptorSets(device, &allocInfo, &newDescriptorSet);
        if (result != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to allocate descriptor set");
        }

        VkWriteDescriptorSetAccelerationStructureKHR asInfo{};
        asInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
        asInfo.accelerationStructureCount = 1;
        asInfo.pAccelerationStructures = &(tlas.handle);

        VkWriteDescriptorSet descriptorWrite{};
        descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrite.dstSet = newDescriptorSet;
        descriptorWrite.dstBinding = 0;
        descriptorWrite.dstArrayElement = 0;
        descriptorWrite.descriptorCount = 1;
        descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        descriptorWrite.pNext = &asInfo;

        vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);

        return newDescriptorSet;
    }

    void buildSBT()
    {
        const uint32_t groupCount = 3; // raygen, miss, hit
        uint32_t handleSize, baseAlignment, handleSizeAligned;
        getHandleSizeAligned(physicalDevice, handleSize, baseAlignment, handleSizeAligned);

        raygenRegion.stride = align_up(handleSizeAligned, baseAlignment);
        raygenRegion.size = raygenRegion.stride; // The size member of pRayGenShaderBindingTable must be equal to its stride member
        missRegion.stride = handleSizeAligned;
        missRegion.size = align_up(1 * missRegion.stride, baseAlignment);
        hitRegion.stride = align_up(handleSize + sizeof(vec4), handleSize);
        hitRegion.size = align_up(1 * hitRegion.stride, baseAlignment);
        std::cout << "ray gen shader: stride = " << raygenRegion.stride << ", size = " << raygenRegion.size << std::endl;
        std::cout << "ray miss shader: stride = " << missRegion.stride << ", size = " << missRegion.size << std::endl;
        std::cout << "ray hit shader: stride = " << hitRegion.stride << ", size = " << hitRegion.size << std::endl;

        // Allocate a buffer for storing the SBT.
        sbtBuffer.size = raygenRegion.size + missRegion.size + hitRegion.size;
        createBuffer(device, physicalDevice, sbtBuffer.size,
                     VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     sbtBuffer.buffer, sbtBuffer.memory);
        sbtBuffer.deviceAddress = getBufferDeviceAddress(device, sbtBuffer.buffer);
        // Find the SBT addresses of each group
        raygenRegion.deviceAddress = sbtBuffer.deviceAddress;
        missRegion.deviceAddress = sbtBuffer.deviceAddress + raygenRegion.size;
        hitRegion.deviceAddress = sbtBuffer.deviceAddress + raygenRegion.size + missRegion.size;

        // Get shader group handles
        std::vector<uint8_t> handles(sbtBuffer.size);
        PFN_vkGetRayTracingShaderGroupHandlesKHR vkGetRayTracingShaderGroupHandlesKHR =
            (PFN_vkGetRayTracingShaderGroupHandlesKHR)vkGetDeviceProcAddr(device, "vkGetRayTracingShaderGroupHandlesKHR");
        vkGetRayTracingShaderGroupHandlesKHR(device, rayPipeline, 0, groupCount, handles.size(), handles.data());
        // Write shader handles to the buffer
        uint8_t *mapped;
        vkMapMemory(device, sbtBuffer.memory, 0, sbtBuffer.size, 0, (void **)&mapped);
        memcpy(mapped, handles.data(), handleSize);
        memcpy(mapped + raygenRegion.size, handles.data() + handleSize, handleSize);
        memcpy(mapped + raygenRegion.size + missRegion.size, handles.data() + 2 * handleSize, handleSize);
        vkUnmapMemory(device, sbtBuffer.memory);

        std::cout << "handleSize = " << handleSize << std::endl;
        std::cout << "handleSizeAligned = " << handleSizeAligned << std::endl;
    }

    void initDescriptor()
    {
        VkWriteDescriptorSetAccelerationStructureKHR tlasDesc{};
        tlasDesc.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
        tlasDesc.accelerationStructureCount = 1;
        tlasDesc.pAccelerationStructures = &tlas.handle;

        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = occlusionBuffer.buffer;
        bufferInfo.offset = 0;
        bufferInfo.range = occlusionBuffer.size;

        VkWriteDescriptorSet descWrites[2] = {};

        descWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descWrites[0].dstSet = descriptorSet;
        descWrites[0].dstBinding = eTlas;
        descWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        descWrites[0].pNext = &tlasDesc;
        descWrites[0].descriptorCount = 1;

        descWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descWrites[1].dstSet = descriptorSet;
        descWrites[1].dstBinding = eOutImage;
        descWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descWrites[1].pBufferInfo = &bufferInfo;
        descWrites[1].descriptorCount = 1;

        vkUpdateDescriptorSets(device, 2, descWrites, 0, nullptr);
    }

    ~RayQueryApp()
    {
        cleanUp();
    }

    void cleanUp()
    {
        vkDestroyCommandPool(device, commandPool, nullptr);
        vkDestroyFence(device, fence, nullptr);

        vkDestroyShaderModule(device, raygenModule, nullptr);
        vkDestroyShaderModule(device, missModule, nullptr);
        vkDestroyShaderModule(device, chitModule, nullptr);

        vkDestroyPipeline(device, rayPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

        // Don't forget acceleration structures if used
        PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR =
            (PFN_vkDestroyAccelerationStructureKHR)vkGetDeviceProcAddr(device, "vkDestroyAccelerationStructureKHR");
        vkDestroyAccelerationStructureKHR(device, tlas.handle, nullptr);
        vkDestroyAccelerationStructureKHR(device, blas.handle, nullptr);

        vkDestroyBuffer(device, vertexBuffer.buffer, nullptr);
        vkFreeMemory(device, vertexBuffer.memory, nullptr);
        vkDestroyBuffer(device, indicesBuffer.buffer, nullptr);
        vkFreeMemory(device, indicesBuffer.memory, nullptr);
        vkDestroyBuffer(device, alignedIndicesBuffer.buffer, nullptr);
        vkFreeMemory(device, alignedIndicesBuffer.memory, nullptr);

        // vkDestroyBuffer(device, txBuffer.buffer, nullptr);
        // vkFreeMemory(device, txBuffer.memory, nullptr);
        // vkDestroyBuffer(device, rayDirsBuffer.buffer, nullptr);
        // vkFreeMemory(device, rayDirsBuffer.memory, nullptr);

        // vkDestroyBuffer(device, occlusionBuffer.buffer, nullptr);
        // vkFreeMemory(device, occlusionBuffer.memory, nullptr);
        // vkDestroyBuffer(device, occlusionStagingBuffer.buffer, nullptr);
        // vkFreeMemory(device, occlusionStagingBuffer.memory, nullptr);

        vkDestroyBuffer(device, scratchBlasBuffer.buffer, nullptr);
        vkFreeMemory(device, scratchBlasBuffer.memory, nullptr);

        vkDestroyBuffer(device, instanceBuffer.buffer, nullptr);
        vkFreeMemory(device, instanceBuffer.memory, nullptr);
        vkDestroyBuffer(device, scratchTlasBuffer.buffer, nullptr);
        vkFreeMemory(device, scratchTlasBuffer.memory, nullptr);

        vkDestroyBuffer(device, sbtBuffer.buffer, nullptr);
        vkFreeMemory(device, sbtBuffer.memory, nullptr);

        vkDestroyBuffer(device, blas.buffer, nullptr);
        vkFreeMemory(device, blas.memory, nullptr);
        vkDestroyBuffer(device, tlas.buffer, nullptr);
        vkFreeMemory(device, tlas.memory, nullptr);

        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
    }

    void runRayQuery(VkCommandBuffer cmdBuf, uint32_t height, uint32_t width)
    {
        // 2.0 Execution
        vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rayPipeline);
        vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                                pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

        vkCmdPushConstants(
            cmdBuf,
            pipelineLayout,
            VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
            0,
            sizeof(objDescData),
            &objDescData);

        PFN_vkCmdTraceRaysKHR vkCmdTraceRaysKHR =
            (PFN_vkCmdTraceRaysKHR)vkGetDeviceProcAddr(device, "vkCmdTraceRaysKHR");
        vkCmdTraceRaysKHR(cmdBuf,
                          &raygenRegion,
                          &missRegion,
                          &hitRegion,
                          &callableRegion,
                          width, height, 1); // Use 30 x 20 threads

        // Add memory barrier AFTER compute/ray tracing shader
        VkBufferMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = occlusionBuffer.buffer; // GPU wrote to this
        barrier.offset = 0;
        barrier.size = VK_WHOLE_SIZE;

        vkCmdPipelineBarrier(
            cmdBuf,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            VK_PIPELINE_STAGE_HOST_BIT,
            0,
            0, nullptr,
            1, &barrier,
            0, nullptr);

        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = 0;
        copyRegion.size = occlusionBuffer.size;
        vkCmdCopyBuffer(cmdBuf, occlusionBuffer.buffer, occlusionStagingBuffer.buffer, 1, &copyRegion);
    }

    std::vector<float> QueryForLOS(std::vector<float> &txs, std::vector<float> &rxs)
    {
        uint32_t height = txs.size() / 3;
        uint32_t width = rxs.size() / 3 / height;
        std::vector<Vertex> vertex_txs(height);
        std::vector<Vertex> vertex_rayDirs(height * width);
        for (int j = 0; j < height; ++j)
        {
            vertex_txs[j] = Vertex(txs[3 * j], txs[3 * j + 1], txs[3 * j + 2]);
            for (int i = 0; i < width; ++i)
            {
                int idx = j * width + i;
                vec3 rayDir = vec3(rxs[3 * idx] - txs[3 * j], rxs[3 * idx + 1] - txs[3 * j + 1], rxs[3 * idx + 2] - txs[3 * j + 2]);
                rayDir = normalize(rayDir);
                vertex_rayDirs[idx] = Vertex(rayDir[0], rayDir[1], rayDir[2]);
            }
        }
        // 1. Command Begin
        VkCommandBuffer cmdBuf = createNewCommandBuffer();
        vkResetFences(device, 1, &fence);
        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(cmdBuf, &beginInfo);

        // 2. Main code
        createTxRaydirOcclusionBuffers(cmdBuf, vertex_txs, vertex_rayDirs);
        initDescriptor();
        runRayQuery(cmdBuf, height, width);

        // 3. Submit command buffer
        vkEndCommandBuffer(cmdBuf);
        VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmdBuf;
        vkQueueSubmit(computeQueue, 1, &submitInfo, fence);
        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
        vkFreeCommandBuffers(device, commandPool, 1, &cmdBuf);

        float *data = nullptr;
        vkMapMemory(device, occlusionStagingBuffer.memory, 0, VK_WHOLE_SIZE, 0, (void **)&data);
        vkUnmapMemory(device, occlusionStagingBuffer.memory);
        std::vector<float> output_mat(height * width * 4);
        memcpy(output_mat.data(), data, sizeof(float) * output_mat.size());

        vkDestroyBuffer(device, txBuffer.buffer, nullptr);
        vkFreeMemory(device, txBuffer.memory, nullptr);
        vkDestroyBuffer(device, rayDirsBuffer.buffer, nullptr);
        vkFreeMemory(device, rayDirsBuffer.memory, nullptr);

        vkDestroyBuffer(device, txStagingBuffer.buffer, nullptr);
        vkFreeMemory(device, txStagingBuffer.memory, nullptr);
        vkDestroyBuffer(device, rayDirsStagingBuffer.buffer, nullptr);
        vkFreeMemory(device, rayDirsStagingBuffer.memory, nullptr);

        vkDestroyBuffer(device, occlusionBuffer.buffer, nullptr);
        vkFreeMemory(device, occlusionBuffer.memory, nullptr);
        vkDestroyBuffer(device, occlusionStagingBuffer.buffer, nullptr);
        vkFreeMemory(device, occlusionStagingBuffer.memory, nullptr);

        // std::vector<float> output_depth(height * width, -1);
        // for (int j = 0; j < height; ++j)
        // {
        //     vec3 this_tx = vec3(txs[3 * j], txs[3 * j + 1], txs[3 * j + 2]);
        //     for (int i = 0; i < width; ++i)
        //     {
        //         int idx = j * width + i;
        //         vec3 this_rx = vec3(rxs[3 * idx], rxs[3 * idx + 1], rxs[3 * idx + 2]);
        //         float tx_rx_depth = length(this_rx - this_tx);
        //         float geom_depth = output_mat[idx * 4 + 3];
        //         if (geom_depth < 0) {
        //             output_depth[idx] = geom_depth;
        //         }
        //     }
        // }

        return output_mat;
    }

    std::vector<float> QueryForNLOS(std::vector<float> &rxs, std::vector<float> &rayDirs)
    {
        uint32_t height = rxs.size() / 3, width = rayDirs.size() / 3 / height;
        std::vector<Vertex> vertex_rxs(height);
        std::vector<Vertex> vertex_rayDirs(height * width);
        for (int idx = 0; idx < height * width; ++idx)
        {
            vec3 rayDir = vec3(rayDirs[3 * idx], rayDirs[3 * idx + 1], rayDirs[3 * idx + 2]);
            rayDir = normalize(rayDir);
            vertex_rayDirs[idx] = Vertex(rayDir[0], rayDir[1], rayDir[2]);
        }
        for (int j = 0; j < height; ++j)
        {
            vertex_rxs[j] = Vertex(rxs[3 * j], rxs[3 * j + 1], rxs[3 * j + 2]);
        }
        // 1. Command Begin
        VkCommandBuffer cmdBuf = createNewCommandBuffer();
        vkResetFences(device, 1, &fence);
        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(cmdBuf, &beginInfo);

        // 2. Main code
        createTxRaydirOcclusionBuffers(cmdBuf, vertex_rxs, vertex_rayDirs);
        initDescriptor();
        runRayQuery(cmdBuf, height, width);

        // 3. Submit command buffer
        vkEndCommandBuffer(cmdBuf);
        VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmdBuf;
        vkQueueSubmit(computeQueue, 1, &submitInfo, fence);
        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
        vkFreeCommandBuffers(device, commandPool, 1, &cmdBuf);

        float *data = nullptr;
        vkMapMemory(device, occlusionStagingBuffer.memory, 0, VK_WHOLE_SIZE, 0, (void **)&data);
        std::vector<float> output_mat(height * width * 4, 4.3);
        memcpy(output_mat.data(), data, sizeof(float) * output_mat.size());
        vkUnmapMemory(device, occlusionStagingBuffer.memory);
        // for (int idx = 0; idx < height * width; ++idx)
        // {
        //     std::cout << "[debug] idx = " << idx << ", depth = " << output_mat[idx * 4 + 3] << std::endl;
        // }

        vkDestroyBuffer(device, txBuffer.buffer, nullptr);
        vkFreeMemory(device, txBuffer.memory, nullptr);
        vkDestroyBuffer(device, rayDirsBuffer.buffer, nullptr);
        vkFreeMemory(device, rayDirsBuffer.memory, nullptr);

        vkDestroyBuffer(device, txStagingBuffer.buffer, nullptr);
        vkFreeMemory(device, txStagingBuffer.memory, nullptr);
        vkDestroyBuffer(device, rayDirsStagingBuffer.buffer, nullptr);
        vkFreeMemory(device, rayDirsStagingBuffer.memory, nullptr);

        vkDestroyBuffer(device, occlusionBuffer.buffer, nullptr);
        vkFreeMemory(device, occlusionBuffer.memory, nullptr);
        vkDestroyBuffer(device, occlusionStagingBuffer.buffer, nullptr);
        vkFreeMemory(device, occlusionStagingBuffer.memory, nullptr);

        return output_mat;
    }
};
#endif