#version 450

// ============================================================================
// OSRS-style Gouraud vertex shader.
//
// WebGL2 -> Vulkan notes:
//  - `layout(set = 0, binding = 0)` replaces WebGL2's
//    gl.uniformBlockBinding / gl.bindBufferBase pair. The (set, binding)
//    address is fixed in the pipeline layout at pipeline creation; nothing
//    is looked up by name at runtime.
//  - This file is compiled to SPIR-V by glslc at BUILD time (see
//    CMakeLists.txt). WebGL2 shipped this source to the driver and compiled
//    it at runtime with gl.compileShader - a stall you paid on first use,
//    on the user's machine, with driver-specific error behavior.
//  - gl_Position lands in Vulkan clip space: +Y down, depth 0..1. The
//    projection matrix already accounts for both (camera.cpp).
// ============================================================================

layout(set = 0, binding = 0) uniform FrameUniforms {
    mat4 view;
    mat4 proj;
    vec4 lightDirection; // xyz: normalized world-space travel direction of the light
    vec4 lightParams;    // x: ambient term, y: diffuse strength
} frame;

// Per-vertex attributes (binding 0, VK_VERTEX_INPUT_RATE_VERTEX).
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
// RGB + opacity; the alpha channel passes through lighting untouched.
layout(location = 2) in vec4 inColor;
// Texture coordinates + array layer; layer < 0 means untextured.
layout(location = 3) in vec3 inUvLayer;

// Per-instance model matrix (binding 1, VK_VERTEX_INPUT_RATE_INSTANCE).
// A mat4 attribute occupies four consecutive locations (4, 5, 6, 7) - the
// C++ side declares it as four vec4 columns.
layout(location = 4) in mat4 inModel;

layout(location = 0) out vec4 vColor;
layout(location = 1) out vec3 vUvLayer;

void main() {
    // The placeholder model (and OSRS gear in general) only uses rotation,
    // translation and uniform scale, so the upper 3x3 of the model matrix
    // is a valid normal transform. Non-uniform scale would need the
    // inverse-transpose instead.
    vec3 worldNormal = normalize(mat3(inModel) * inNormal);

    // The OSRS look: lighting is evaluated HERE, once per vertex, and the
    // rasterizer interpolates the resulting color across the triangle
    // (Gouraud shading). A modern renderer would pass the normal through
    // and light per fragment; per-vertex evaluation on low-poly geometry is
    // what produces the characteristic soft facet gradients. Flat ambient
    // plus one directional light - same terms the original client uses.
    float nDotL = max(dot(worldNormal, -frame.lightDirection.xyz), 0.0);
    float brightness = frame.lightParams.x + frame.lightParams.y * nDotL;

    // Clamp per vertex, before interpolation, so overbright faces saturate
    // uniformly instead of skewing the gradient across the triangle.
    vColor = vec4(min(inColor.rgb * brightness, vec3(1.0)), inColor.a);
    // The layer is constant across a face, so interpolation is a no-op.
    vUvLayer = inUvLayer;

    gl_Position = frame.proj * frame.view * inModel * vec4(inPosition, 1.0);
}
