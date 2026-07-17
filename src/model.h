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
    glm::vec3 color;
    // Texture coordinates + array layer for textured faces (world mode).
    // z < 0 means untextured: the fragment shader uses the vertex color
    // alone, so models with no textures (like the placeholder) ignore this.
    glm::vec3 uvLayer{0.0f, 0.0f, -1.0f};
};

// A model is exactly what the GPU consumes: a vertex array and a uint32
// triangle-list index array (three indices per face).
//
// Flat-shaded faces duplicate their three vertices so each corner can carry
// the face normal and face color; smooth (Gouraud) faces share vertices
// through the index buffer. OSRS models mix both per face, which is why the
// interface is indexed even though the procedural placeholder below happens
// to be fully flat-shaded.
struct Model {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
};

// ---------------------------------------------------------------------------
// INTEGRATION POINT for a real OSRS model decoder.
//
// The renderer is completely agnostic about where the model comes from; it
// only sees the Model struct above. To display real cache models, replace
// the body of loadModel() with a decoder that fills in the same contract:
//
//   - triangle list, CCW winding when viewed from outside
//   - +Y up, ground plane at y = 0, roughly 1 unit ~ 1 game "tile-ish" meter
//   - colors as linear RGB (OSRS's 16-bit HSL palette entries converted)
//   - flat faces: duplicate vertices with the face normal
//   - smooth faces: shared vertices with area-weighted vertex normals
//
// The built-in placeholder is a procedurally generated low-poly tree so the
// project runs with zero assets.
// ---------------------------------------------------------------------------
Model loadModel();
