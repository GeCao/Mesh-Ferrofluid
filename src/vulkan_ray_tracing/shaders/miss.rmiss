#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_ray_query : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "raycommon.glsl"

// clang-format off
layout(location = 0) rayPayloadInEXT vec4 prd;

layout(std430, buffer_reference, scalar) buffer Vertices {Vertex v[]; }; // Positions of an object
layout(std430, buffer_reference, scalar) buffer Indices {Indice i[]; }; // Triangle indices
layout(set = 0, binding = eTlas) uniform accelerationStructureEXT topLevelAS;
layout(push_constant) uniform ObjDesc_ { ObjDesc i[]; } objDesc;
// clang-format on

void main()
{
  prd = vec4(-1, 0, 0, -1); // No occlusion -> w=-1, indicates no geometry detected
}
