#pragma once

#include <glm/glm.hpp>

#include <cstdint>
#include <vector>

// One vertex of a renderable model.
//
// This mirrors the vertex layout of the XRSPS WebGL2 renderer: OSRS models
// carry no texture coordinates on most geometry - the look comes from a
// per-face 16-bit HSL palette color plus per-vertex lighting. Colors are
// stored here as linear RGB in [0, 1].
//
// The struct's exact byte layout matters: renderer.cpp describes it to the
// pipeline field-by-field with offsetof(). In WebGL2 the same contract was
// expressed with gl.vertexAttribPointer(size, type, stride, offset) - the
// difference is that Vulkan validates it against the shader's declared
// inputs at pipeline creation instead of at draw time.
struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    // RGB color plus opacity: 1 = opaque, lower values are the game's
    // per-face translucency (spider webs, waterfalls, portals).
    glm::vec4 color;
    // Texture coordinates + array layer for textured faces. z < 0 means
    // untextured: the fragment shader uses the vertex color alone.
    glm::vec3 uvLayer{0.0f, 0.0f, -1.0f};
};

// A model is exactly what the GPU consumes: a vertex array and a uint32
// triangle-list index array (three indices per face).
//
// Flat-shaded faces duplicate their three vertices so each corner can carry
// the face normal and face color; smooth (Gouraud) faces share vertices
// through the index buffer. OSRS models mix both per face.
//
// Translucent faces go into a separate index list over the same vertex
// array: the renderer draws `indices` with the opaque pipeline first, then
// `alphaIndices` with blending on and depth writes off.
struct Model {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<uint32_t> alphaIndices;
};
