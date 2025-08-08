// main.cpp - Vulkan Ray Query Example (Headless, No GUI)
// Output: Occlusion matrix (30x20) for fixed Txs and Rxs testing against a box

#include <vulkan/vulkan.h>
#include <iostream>
#include <vector>
#include <array>
#include <cassert>
#include <cstring>
#include <glm/glm.hpp>

#include "ray_query_App.hpp"

int main()
{
    std::vector<float> vertices{-1, -1, -1,
                                +1, -1, -1,
                                +1, +1, -1,
                                -1, +1, -1,
                                -1, -1, +1,
                                +1, -1, +1,
                                +1, +1, +1,
                                -1, +1, +1};
    std::vector<uint32_t> indices{0, 1, 2, 2, 3, 0,
                                  4, 5, 6, 6, 7, 4,
                                  0, 1, 5, 5, 4, 0,
                                  2, 3, 7, 7, 6, 2,
                                  0, 3, 7, 7, 4, 0,
                                  1, 2, 6, 6, 5, 1};
    std::vector<float> txs{-2, 0, 0, +2, 0, 0, 0, -2, 0, 0, +2, 0, 0, 0, -2, 0, 0, +2};
    std::vector<float> rxs{-5, 0, 0, +5, 0, 0, 0, -5, 0, 0, +5, 0, 0, 0, -5, 0, 0, +5};
    uint32_t height = txs.size() / 3;
    uint32_t width = rxs.size() / 3;
    rxs.resize(height * width * 3);
    for (int tx = 1; tx < height; ++tx)
    {
        for (int rx = 0; rx < width; ++rx)
        {
            int idx = tx * width + rx;
            rxs[3 * idx] = rxs[3 * rx];
            rxs[3 * idx + 1] = rxs[3 * rx + 1];
            rxs[3 * idx + 2] = rxs[3 * rx + 2];
        }
    }

    std::string raygenShaderPath = "spv/raygen.spv";
    std::string missShaderPath = "spv/miss.spv";
    std::string chitShaderPath = "spv/closesthit.spv";
    RayQueryApp app(vertices, indices, raygenShaderPath, missShaderPath, chitShaderPath);
    std::vector<float> occlusionMatrix = app.QueryForLOS(txs, rxs);
    // Now `data` points to a 30×20 float matrix in row-major order
    for (int tx = 0; tx < height; ++tx)
    {
        for (int rx = 0; rx < width; ++rx)
        {
            float value_r = occlusionMatrix[tx * width * 4 + rx * 4 + 0];
            float value_g = occlusionMatrix[tx * width * 4 + rx * 4 + 1];
            float value_b = occlusionMatrix[tx * width * 4 + rx * 4 + 2];
            float value_a = occlusionMatrix[tx * width * 4 + rx * 4 + 3];
            std::cout << "Occlusion[" << tx << "][" << rx << "] = " << value_r << ", " << value_g << ", " << value_b << ", " << value_a << std::endl;
        }
    }
    return EXIT_SUCCESS;
}