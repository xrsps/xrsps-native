#pragma once

// World streaming: turns real OSRS cache map data into renderable chunks.
// This is a faithful port of the terrain + scenery pipeline of the XRSPS
// WebGL2 scene builder (itself matching the deobfuscated client): tile
// decode, procedural height noise, underlay color blending, overlay tile
// shapes, per-vertex hillshade lighting, the HSL palette, loc placements,
// the classic/V2/V3 model formats, and sampled sprite textures.
//
// One WorldStreamer holds everything global (configs, palette, textures,
// lit-model cache); buildSquare() then produces one 64x64 map square's
// geometry in ABSOLUTE world coordinates (1 unit = 1 tile, +Y up, +X east,
// -Z north; viewer x = world tile x, viewer z = -world tile y), so squares
// can be streamed in and out around the camera independently. Each square
// is built with its 8 neighbours decoded as context, which keeps color
// blending and lighting seamless across chunk borders.
//
// Vertex colors arrive fully lit (the game bakes its lighting into terrain
// and model colors at scene-build time), so the renderer draws them with
// lighting disabled (ambient 1, diffuse 0).

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "model.h"

class WorldStreamer {
public:
    WorldStreamer();
    ~WorldStreamer();

    // Loads the cache and everything square-independent: floor/loc configs,
    // the color palette, and the texture array. Throws std::runtime_error
    // with a descriptive message on failure.
    void open(const std::string& cachePath);

    // Game textures as 128x128 RGBA layers (sRGB bytes, alpha 0 = cutout);
    // hand these to the Renderer once. Vertex uvLayer.z indexes this array.
    const std::vector<std::vector<uint8_t>>& textures() const;

    // Builds one map square's terrain + scenery. Returns false if the cache
    // has no data for that square (open sea, unreleased areas).
    bool buildSquare(int mapX, int mapY, Model& out);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
