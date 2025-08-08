#ifndef COMMON_HOST_DEVICE
#define COMMON_HOST_DEVICE

#ifdef __cplusplus
#include <glm/glm.hpp>
// GLSL Type
using ivec3 = glm::ivec3;
using vec2 = glm::vec2;
using vec3 = glm::vec3;
using vec4 = glm::vec4;
using mat4 = glm::mat4;
using uint = unsigned int;
#endif

// clang-format off
#ifdef __cplusplus // Descriptor binding helper for C++ and GLSL
 #define START_BINDING(a) enum a {
 #define END_BINDING() }
#else
 #define START_BINDING(a)  const uint
 #define END_BINDING() 
#endif

START_BINDING(SceneBindings)
  eGlobals  = 0,  // Global uniform containing camera matrices
  eObjDescs = 1,  // Access to the object descriptions
  eTextures = 2   // Access to textures
END_BINDING();

START_BINDING(RtxBindings)
  eTlas     = 0,  // Top-level acceleration structure
  eOutImage = 1   // Ray tracer output image
END_BINDING();
// clang-format on

// Information of a obj model when referenced in a shader
struct ObjDesc
{
    uint64_t vertexAddress; // Address of the Vertex buffer
    uint64_t indexAddress;  // Address of the index buffer
    uint64_t txAddress;     // Address of the Tx buffer
    uint64_t rayDirAddress; // Address of the Raydir buffer
};

struct Vertex // See ObjLoader, copy of VertexObj, could be compressed for device
{
    vec3 pos;
    // vec3 nrm;
    // vec3 color;
    // vec2 texCoord;
    float pad; // for 64-byte alignment

#ifdef __cplusplus
    Vertex()
    {
        pos = vec3(0, 0, 0);
    }
    Vertex(float x, float y, float z)
    {
        pos = vec3(x, y, z);
    }
    Vertex(float x, float y, float z, float pad_)
    {
        pos = vec3(x, y, z);
        pad = pad_;
    }
    Vertex &operator=(const Vertex &other) = default;
#endif
};

struct Indice
{
    ivec3 triangle_index;
    int pad; // for 64-byte alignment

#ifdef __cplusplus
    Indice()
    {
        triangle_index = ivec3(0, 0, 0);
    }
    Indice(int i1, int i2, int i3)
    {
        triangle_index = ivec3(i1, i2, i3);
    }
    Indice(int i1, int i2, int i3, int pad_)
    {
        triangle_index = ivec3(i1, i2, i3);
        pad = pad_;
    }
    Indice &operator=(const Indice &other) = default;
#endif
};

#endif
