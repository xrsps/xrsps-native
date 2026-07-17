#version 450

// With Gouraud shading all lighting work happened in the vertex shader, and
// the color arrives here already interpolated by the rasterizer. Textured
// faces (world mode: water, roads, leaves, roofs) sample a 2D array texture
// and use the vertex color as a per-vertex shade multiplier; palette entry 0
// pixels have alpha 0 and are cut out, which is how the game does leaf and
// fence transparency without blending or sorting.

layout(set = 0, binding = 1) uniform sampler2DArray uTextures;

layout(location = 0) in vec3 vColor;
layout(location = 1) in vec3 vUvLayer;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 color = vColor;
    if (vUvLayer.z >= 0.0) {
        vec4 texel = texture(uTextures, vUvLayer);
        if (texel.a < 0.5) {
            discard;
        }
        color = texel.rgb * vColor;
    }
    // The swapchain image is sRGB (VK_FORMAT_B8G8R8A8_SRGB where available),
    // so this value is linear and the hardware gamma-encodes it on store -
    // the same division of labor as a WebGL2 canvas backbuffer.
    outColor = vec4(color, 1.0);
}
