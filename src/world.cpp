#include "world.h"

#include "cache.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// Everything in this file is integer-exact against the reference client
// implementation (via the XRSPS TypeScript port): the height noise, the
// blending averages, the HSL palette, the lighting - all of it uses the same
// truncating integer arithmetic, so the produced colors match the game's
// pixel for pixel (up to our lack of ground textures, which are approximated
// by each texture's average color, the same substitution the game itself
// uses for the minimap).
namespace {

// --- world constants ---------------------------------------------------------
constexpr int kMapSquareSize = 64;   // tiles per map square edge
constexpr int kMaxLevels = 4;        // ground + 3 upper planes
constexpr int kTileUnits = 128;      // fixed-point units per tile edge
constexpr int kHeightBasis = 8;      // height-map units per stored height step
constexpr int kLevelHeight = 240;    // units between vertical planes
constexpr int kBlendRadius = 5;      // underlay color blend kernel radius
constexpr int32_t kInvalidHsl = 12345678;

constexpr int kConfigsIndex = 2;
constexpr int kMapsIndex = 5;
constexpr int kSpritesIndex = 8;
constexpr int kTexturesIndex = 9;
constexpr int kUnderlaysGroup = 1;
constexpr int kOverlaysGroup = 4;

// JS Math.imul: 32-bit wrapping multiply.
int32_t imul(int32_t a, int32_t b) {
    return static_cast<int32_t>(static_cast<uint32_t>(a) * static_cast<uint32_t>(b));
}

// --- procedural height noise -------------------------------------------------
// Tiles whose map data carries no explicit height fall back to this fixed
// Perlin-style noise, seeded by absolute world coordinates - which is why
// the scene builder needs to know where in the world it is building.

const std::array<int32_t, 2048>& cosineTable() {
    static const auto table = [] {
        std::array<int32_t, 2048> t{};
        for (int i = 0; i < 2048; ++i) {
            t[static_cast<size_t>(i)] =
                static_cast<int32_t>(65536.0 * std::cos(i * (3.141592653589793 * 2.0 / 2048.0)));
        }
        return t;
    }();
    return table;
}

int32_t interpolateCos(int32_t a, int32_t b, int32_t x, int32_t freq) {
    const int32_t t = (65536 - cosineTable()[static_cast<size_t>((x * 1024) / freq)]) >> 1;
    return ((t * b) >> 16) + (((65536 - t) * a) >> 16);
}

int32_t noise(int32_t x, int32_t y) {
    int32_t n = y * 57 + x;
    n = (n << 13) ^ n;
    const int32_t n2 =
        (imul(n, imul(imul(n, n), 15731) + 789221) + 1376312589) & 0x7fffffff;
    return (n2 >> 19) & 0xff;
}

int32_t smoothedNoise(int32_t x, int32_t y) {
    const int32_t corners =
        noise(x - 1, y - 1) + noise(x + 1, y - 1) + noise(x - 1, y + 1) + noise(x + 1, y + 1);
    const int32_t sides = noise(x - 1, y) + noise(x + 1, y) + noise(x, y - 1) + noise(x, y + 1);
    return corners / 16 + sides / 8 + noise(x, y) / 4;
}

int32_t interpolateNoise(int32_t x, int32_t y, int32_t freq) {
    const int32_t intX = x / freq;
    const int32_t fracX = x & (freq - 1);
    const int32_t intY = y / freq;
    const int32_t fracY = y & (freq - 1);
    const int32_t v1 = smoothedNoise(intX, intY);
    const int32_t v2 = smoothedNoise(intX + 1, intY);
    const int32_t v3 = smoothedNoise(intX, intY + 1);
    const int32_t v4 = smoothedNoise(intX + 1, intY + 1);
    const int32_t i1 = interpolateCos(v1, v2, fracX, freq);
    const int32_t i2 = interpolateCos(v3, v4, fracX, freq);
    return interpolateCos(i1, i2, fracY, freq);
}

int32_t generateHeight(int32_t x, int32_t y) {
    int32_t n = interpolateNoise(x + 45365, y + 91923, 4) - 128 +
                ((interpolateNoise(x + 10294, y + 37821, 2) - 128) >> 1) +
                ((interpolateNoise(x, y, 1) - 128) >> 2);
    n = static_cast<int32_t>(0.3 * n) + 35;
    return std::clamp(n, 10, 60);
}

// --- HSL color math ------------------------------------------------------------

int32_t packHsl(int32_t hue, int32_t saturation, int32_t lightness) {
    if (lightness > 179) saturation /= 2;
    if (lightness > 192) saturation /= 2;
    if (lightness > 217) saturation /= 2;
    if (lightness > 243) saturation /= 2;
    return ((saturation / 32) << 7) + ((hue / 4) << 10) + (lightness / 2);
}

int32_t mixHsl(int32_t a, int32_t b) {
    if (a == kInvalidHsl || b == kInvalidHsl) return kInvalidHsl;
    if (a == -1) return b;
    if (b == -1) return a;
    const int32_t hue = (((a >> 10) & 0x3f) + ((b >> 10) & 0x3f)) >> 1;
    const int32_t sat = (((a >> 7) & 0x7) + ((b >> 7) & 0x7)) >> 1;
    const int32_t light = ((a & 0x7f) + (b & 0x7f)) >> 1;
    return (hue << 10) + (sat << 7) + light;
}

int32_t adjustUnderlayLight(int32_t hsl, int32_t light) {
    if (hsl == -1) return kInvalidHsl;
    light = ((hsl & 127) * light) >> 7;
    light = std::clamp(light, 2, 126);
    return (hsl & 0xff80) + light;
}

int32_t adjustOverlayLight(int32_t hsl, int32_t light) {
    if (hsl == -2) return kInvalidHsl;
    if (hsl == -1) return std::clamp(light, 2, 126);
    light = ((hsl & 127) * light) >> 7;
    light = std::clamp(light, 2, 126);
    return (hsl & 0xff80) + light;
}

int32_t brightenRgb(int32_t rgb, double brightness) {
    const double r = std::pow(((rgb >> 16) & 0xff) / 256.0, brightness);
    const double g = std::pow(((rgb >> 8) & 0xff) / 256.0, brightness);
    const double b = std::pow((rgb & 0xff) / 256.0, brightness);
    return (static_cast<int32_t>(r * 256.0) << 16) |
           (static_cast<int32_t>(g * 256.0) << 8) |
           static_cast<int32_t>(b * 256.0);
}

// The game's packed-HSL -> RGB palette (brightness 0.8, the client default).
// Index = 16-bit packed HSL; value = 0xRRGGBB.
std::vector<int32_t> buildPalette(double brightness) {
    std::vector<int32_t> palette(65536, 0);
    size_t paletteIndex = 0;
    for (int hs = 0; hs < 512; ++hs) {
        const double hue = (hs >> 3) / 64.0 + 0.0078125;
        const double sat = (hs & 7) / 8.0 + 0.0625;
        for (int li = 0; li < 128; ++li) {
            const double lightness = li / 128.0;
            double r = lightness;
            double g = lightness;
            double b = lightness;
            if (sat != 0.0) {
                double q;
                if (lightness < 0.5) {
                    q = lightness * (1.0 + sat);
                } else {
                    q = lightness + sat - lightness * sat;
                }
                const double p = 2.0 * lightness - q;
                auto channel = [&](double h) {
                    if (h > 1.0) h -= 1.0;
                    if (h < 0.0) h += 1.0;
                    if (6.0 * h < 1.0) return p + (q - p) * 6.0 * h;
                    if (2.0 * h < 1.0) return q;
                    if (3.0 * h < 2.0) return p + (q - p) * (2.0 / 3.0 - h) * 6.0;
                    return p;
                };
                r = channel(hue + 1.0 / 3.0);
                g = channel(hue);
                b = channel(hue - 1.0 / 3.0);
            }
            int32_t rgb = (static_cast<int32_t>(r * 256.0) << 16) +
                          (static_cast<int32_t>(g * 256.0) << 8) +
                          static_cast<int32_t>(b * 256.0);
            rgb = brightenRgb(rgb, brightness);
            if (rgb == 0) rgb = 1;
            if (paletteIndex < palette.size()) {
                palette[paletteIndex++] = rgb;
            }
        }
    }
    return palette;
}

// --- floor configs -------------------------------------------------------------
// Underlays (config group 1) and overlays (group 4) both derive HSL fields
// from a packed RGB via the same conversion; the difference is what the
// scene builder does with them (blend kernel vs direct color).

struct HslParts {
    int32_t hue256 = 0;     // hue scaled to 0..255 (overlay packHsl input)
    int32_t saturation = 0; // 0..255
    int32_t lightness = 0;  // 0..255
    int32_t hueMultiplier = 1;
    int32_t hueBlend = 0;   // hueMultiplier * hue fraction (underlay blending)
};

HslParts rgbToHslParts(int32_t rgb) {
    const double r = ((rgb >> 16) & 0xff) / 256.0;
    const double g = ((rgb >> 8) & 0xff) / 256.0;
    const double b = (rgb & 0xff) / 256.0;
    const double minRgb = std::min({r, g, b});
    const double maxRgb = std::max({r, g, b});

    double hue = 0.0;
    double sat = 0.0;
    const double light = (minRgb + maxRgb) / 2.0;
    if (minRgb != maxRgb) {
        if (light < 0.5) sat = (maxRgb - minRgb) / (minRgb + maxRgb);
        if (light >= 0.5) sat = (maxRgb - minRgb) / (2.0 - maxRgb - minRgb);
        if (maxRgb == r) {
            hue = (g - b) / (maxRgb - minRgb);
        } else if (maxRgb == g) {
            hue = 2.0 + (b - r) / (maxRgb - minRgb);
        } else {
            hue = 4.0 + (r - g) / (maxRgb - minRgb);
        }
    }
    hue /= 6.0;

    HslParts parts;
    parts.hue256 = static_cast<int32_t>(hue * 256.0);
    parts.saturation = std::clamp(static_cast<int32_t>(sat * 256.0), 0, 255);
    parts.lightness = std::clamp(static_cast<int32_t>(light * 256.0), 0, 255);
    parts.hueMultiplier = static_cast<int32_t>(
        512.0 * (sat * (light > 0.5 ? 1.0 - light : light)));
    if (parts.hueMultiplier < 1) parts.hueMultiplier = 1;
    parts.hueBlend = static_cast<int32_t>(parts.hueMultiplier * hue);
    return parts;
}

struct UnderlayType {
    int32_t rgb = 0;
    HslParts hsl;

    void decode(js5::ByteReader& r) {
        while (r.remaining() > 0) {
            const uint8_t opcode = r.u8();
            if (opcode == 0) break;
            if (opcode == 1) {
                rgb = static_cast<int32_t>(r.u24());
            } else if (opcode == 2 || opcode == 3) {
                r.u16(); // texture id / texture size (unused for terrain color)
            } else if (opcode == 4 || opcode == 5) {
                // shadow flags, no payload
            } else {
                throw std::runtime_error("underlay: unknown opcode " +
                                         std::to_string(opcode));
            }
        }
        hsl = rgbToHslParts(rgb);
    }
};

struct OverlayType {
    int32_t primaryRgb = 0;
    int32_t textureId = -1;
    int32_t secondaryRgb = -1;
    bool hideUnderlay = true;
    HslParts hsl;

    void decode(js5::ByteReader& r) {
        while (r.remaining() > 0) {
            const uint8_t opcode = r.u8();
            if (opcode == 0) break;
            switch (opcode) {
                case 1: primaryRgb = static_cast<int32_t>(r.u24()); break;
                case 2: textureId = r.u8(); break;
                case 3:
                    textureId = r.u16();
                    if (textureId == 0xffff) textureId = -1;
                    break;
                case 5: hideUnderlay = false; break;
                case 7: secondaryRgb = static_cast<int32_t>(r.u24()); break;
                case 8: break;
                case 9: r.u16(); break;  // texture size
                case 10: break;          // shadowing flag
                case 11: r.u8(); break;  // texture brightness
                case 12: break;          // blend texture flag
                case 13: r.u24(); break; // underwater color
                case 14: r.u8(); break;  // water opacity
                case 15: r.u16(); break; // secondary texture
                case 16: r.u8(); break;
                default:
                    throw std::runtime_error("overlay: unknown opcode " +
                                             std::to_string(opcode));
            }
        }
        hsl = rgbToHslParts(primaryRgb);
    }
};

// --- tile shape templates -------------------------------------------------------
// Overlay tiles are not plain quads: shape 1..12 describes how the tile is
// carved between overlay and underlay (straight splits, corners, slivers),
// using up to 6 of 16 template vertex positions on a quarter-tile grid.
// These two tables ARE the shapes; everything else is bookkeeping.

const std::vector<std::vector<int>> kShapeVertexIndices = {
    {1, 3, 5, 7},
    {1, 3, 5, 7},
    {1, 3, 5, 7},
    {1, 3, 5, 7, 6},
    {1, 3, 5, 7, 6},
    {1, 3, 5, 7, 6},
    {1, 3, 5, 7, 6},
    {1, 3, 5, 7, 2, 6},
    {1, 3, 5, 7, 2, 8},
    {1, 3, 5, 7, 2, 8},
    {1, 3, 5, 7, 11, 12},
    {1, 3, 5, 7, 11, 12},
    {1, 3, 5, 7, 13, 14},
};

// Runs of 4: [isOverlay, vertexA, vertexB, vertexC].
const std::vector<std::vector<int>> kShapeFaces = {
    {0, 1, 2, 3, 0, 0, 1, 3},
    {1, 1, 2, 3, 1, 0, 1, 3},
    {0, 1, 2, 3, 1, 0, 1, 3},
    {0, 0, 1, 2, 0, 0, 2, 4, 1, 0, 4, 3},
    {0, 0, 1, 4, 0, 0, 4, 3, 1, 1, 2, 4},
    {0, 0, 4, 3, 1, 0, 1, 2, 1, 0, 2, 4},
    {0, 1, 2, 4, 1, 0, 1, 4, 1, 0, 4, 3},
    {0, 4, 1, 2, 0, 4, 2, 5, 1, 0, 4, 5, 1, 0, 5, 3},
    {0, 4, 1, 2, 0, 4, 2, 3, 0, 4, 3, 5, 1, 0, 4, 5},
    {0, 0, 4, 5, 1, 4, 1, 2, 1, 4, 2, 3, 1, 4, 3, 5},
    {0, 0, 1, 5, 0, 1, 4, 5, 0, 1, 2, 4, 1, 0, 5, 3, 1, 5, 4, 3, 1, 4, 2, 3},
    {1, 0, 1, 5, 1, 1, 4, 5, 1, 1, 2, 4, 0, 0, 5, 3, 0, 5, 4, 3, 0, 4, 2, 3},
    {1, 0, 5, 4, 1, 0, 1, 5, 0, 0, 4, 3, 0, 4, 5, 3, 0, 5, 2, 3, 0, 1, 2, 5},
};

// --- the scene ------------------------------------------------------------------

struct Scene {
    int sizeX;
    int sizeY;
    std::vector<int32_t> heights, underlays, overlays, shapes, rotations, flags;

    Scene(int sx, int sy) : sizeX(sx), sizeY(sy) {
        const size_t n = static_cast<size_t>(kMaxLevels) * sx * sy;
        heights.assign(n, 0);
        underlays.assign(n, 0);
        overlays.assign(n, 0);
        shapes.assign(n, 0);
        rotations.assign(n, 0);
        flags.assign(n, 0);
    }

    size_t idx(int level, int x, int y) const {
        return (static_cast<size_t>(level) * sizeX + static_cast<size_t>(x)) * sizeY +
               static_cast<size_t>(y);
    }
    int32_t& at(std::vector<int32_t>& v, int level, int x, int y) {
        return v[idx(level, x, y)];
    }
    int32_t get(const std::vector<int32_t>& v, int level, int x, int y) const {
        return v[idx(level, x, y)];
    }
};

// Map terrain file decode ("m" groups). One stream covers all 4 levels of a
// 64x64 square; each tile is a small opcode loop. Build 209+ widened the
// values to 16 bits (the "new" format; this port assumes it).
void decodeTerrainSquare(Scene& scene, const std::vector<uint8_t>& data, int offsetX,
                         int offsetY, int baseAbsX, int baseAbsY) {
    js5::ByteReader r(data);
    for (int level = 0; level < kMaxLevels; ++level) {
        for (int lx = 0; lx < kMapSquareSize; ++lx) {
            for (int ly = 0; ly < kMapSquareSize; ++ly) {
                const int x = lx + offsetX;
                const int y = ly + offsetY;
                while (true) {
                    const uint16_t v = r.u16();
                    if (v == 0) {
                        if (level == 0) {
                            // No explicit height: fixed world-seeded noise.
                            scene.at(scene.heights, 0, x, y) =
                                -generateHeight(baseAbsX + x + 932731,
                                                baseAbsY + y + 556238) *
                                kHeightBasis;
                        } else {
                            scene.at(scene.heights, level, x, y) =
                                scene.get(scene.heights, level - 1, x, y) - kLevelHeight;
                        }
                        break;
                    }
                    if (v == 1) {
                        int32_t height = r.u8();
                        if (height == 1) height = 0;
                        if (level == 0) {
                            scene.at(scene.heights, 0, x, y) = -height * kHeightBasis;
                        } else {
                            scene.at(scene.heights, level, x, y) =
                                scene.get(scene.heights, level - 1, x, y) -
                                height * kHeightBasis;
                        }
                        break;
                    }
                    if (v <= 49) {
                        scene.at(scene.overlays, level, x, y) = r.i16();
                        scene.at(scene.shapes, level, x, y) = (v - 2) / 4;
                        scene.at(scene.rotations, level, x, y) = (v - 2) & 3;
                    } else if (v <= 81) {
                        scene.at(scene.flags, level, x, y) = v - 49;
                    } else {
                        scene.at(scene.underlays, level, x, y) = v - 81;
                    }
                }
            }
        }
    }
}

// Per-vertex hillshade: a fixed directional light dotted against normals
// derived from the height field, in the client's exact integer arithmetic.
// The result is a per-tile-corner "light" scalar that the HSL adjust bakes
// into the final color - terrain lighting is entirely precomputed.
std::vector<int32_t> calculateTileLights(const Scene& scene, int level) {
    std::vector<int32_t> lights(static_cast<size_t>(scene.sizeX) * scene.sizeY, 0);

    const int32_t lightMagnitude =
        static_cast<int32_t>(std::sqrt(50.0 * 50.0 + 10.0 * 10.0 + 50.0 * 50.0));
    const int32_t lightIntensity = (lightMagnitude * 768) >> 8;

    for (int x = 1; x < scene.sizeX - 1; ++x) {
        for (int y = 1; y < scene.sizeY - 1; ++y) {
            const int32_t dx = scene.get(scene.heights, level, x + 1, y) -
                               scene.get(scene.heights, level, x - 1, y);
            const int32_t dy = scene.get(scene.heights, level, x, y + 1) -
                               scene.get(scene.heights, level, x, y - 1);
            const int32_t len = static_cast<int32_t>(std::sqrt(
                static_cast<double>(dy) * dy + static_cast<double>(dx) * dx + 65536.0));
            const int32_t nx = (dx << 8) / len;
            const int32_t ny = 65536 / len;
            const int32_t nz = (dy << 8) / len;
            const int32_t dot = nx * -50 + ny * -10 + nz * -50;
            // (No occlusion term: this viewer loads no walls or roofs yet.)
            lights[static_cast<size_t>(x) * scene.sizeY + y] = dot / lightIntensity + 96;
        }
    }
    return lights;
}

// Underlay color blending: a (2r+1)^2 box average over each component of the
// neighbours' HSL, done with running column/row sums so it stays O(n).
// This smear is why OSRS grass fades smoothly into dirt.
std::vector<int32_t> blendUnderlays(
    const Scene& scene, int level,
    const std::unordered_map<int32_t, UnderlayType>& underlayTypes,
    const UnderlayType& defaultUnderlay) {
    std::vector<int32_t> colors(static_cast<size_t>(scene.sizeX) * scene.sizeY, -1);

    auto underlay = [&](int32_t id) -> const UnderlayType& {
        auto it = underlayTypes.find(id);
        return it == underlayTypes.end() ? defaultUnderlay : it->second;
    };

    const int maxSize = std::max(scene.sizeX, scene.sizeY);
    std::vector<int32_t> hues(static_cast<size_t>(maxSize), 0);
    std::vector<int32_t> sats(static_cast<size_t>(maxSize), 0);
    std::vector<int32_t> light(static_cast<size_t>(maxSize), 0);
    std::vector<int32_t> mul(static_cast<size_t>(maxSize), 0);
    std::vector<int32_t> num(static_cast<size_t>(maxSize), 0);

    for (int xi = -kBlendRadius; xi < scene.sizeX + kBlendRadius; ++xi) {
        for (int yi = 0; yi < scene.sizeY; ++yi) {
            const int xEast = xi + kBlendRadius;
            if (xEast >= 0 && xEast < scene.sizeX) {
                const int32_t id = scene.get(scene.underlays, level, xEast, yi);
                if (id > 0) {
                    const UnderlayType& type = underlay(id - 1);
                    hues[static_cast<size_t>(yi)] += type.hsl.hueBlend;
                    sats[static_cast<size_t>(yi)] += type.hsl.saturation;
                    light[static_cast<size_t>(yi)] += type.hsl.lightness;
                    mul[static_cast<size_t>(yi)] += type.hsl.hueMultiplier;
                    num[static_cast<size_t>(yi)]++;
                }
            }
            const int xWest = xi - kBlendRadius;
            if (xWest >= 0 && xWest < scene.sizeX) {
                const int32_t id = scene.get(scene.underlays, level, xWest, yi);
                if (id > 0) {
                    const UnderlayType& type = underlay(id - 1);
                    hues[static_cast<size_t>(yi)] -= type.hsl.hueBlend;
                    sats[static_cast<size_t>(yi)] -= type.hsl.saturation;
                    light[static_cast<size_t>(yi)] -= type.hsl.lightness;
                    mul[static_cast<size_t>(yi)] -= type.hsl.hueMultiplier;
                    num[static_cast<size_t>(yi)]--;
                }
            }
        }

        if (xi < 0 || xi >= scene.sizeX) continue;

        int32_t runningHues = 0;
        int32_t runningSat = 0;
        int32_t runningLight = 0;
        int32_t runningMultiplier = 0;
        int32_t runningNumber = 0;

        for (int yi = -kBlendRadius; yi < scene.sizeY + kBlendRadius; ++yi) {
            const int yNorth = yi + kBlendRadius;
            if (yNorth >= 0 && yNorth < scene.sizeY) {
                runningHues += hues[static_cast<size_t>(yNorth)];
                runningSat += sats[static_cast<size_t>(yNorth)];
                runningLight += light[static_cast<size_t>(yNorth)];
                runningMultiplier += mul[static_cast<size_t>(yNorth)];
                runningNumber += num[static_cast<size_t>(yNorth)];
            }
            const int ySouth = yi - kBlendRadius;
            if (ySouth >= 0 && ySouth < scene.sizeY) {
                runningHues -= hues[static_cast<size_t>(ySouth)];
                runningSat -= sats[static_cast<size_t>(ySouth)];
                runningLight -= light[static_cast<size_t>(ySouth)];
                runningMultiplier -= mul[static_cast<size_t>(ySouth)];
                runningNumber -= num[static_cast<size_t>(ySouth)];
            }

            if (yi < 0 || yi >= scene.sizeY) continue;
            if (scene.get(scene.underlays, level, xi, yi) <= 0) continue;

            const int32_t avgHue = runningHues * 256 / runningMultiplier;
            const int32_t avgSat = runningSat / runningNumber;
            const int32_t avgLight = runningLight / runningNumber;
            colors[static_cast<size_t>(xi) * scene.sizeY + yi] =
                packHsl(avgHue, avgSat, avgLight);
        }
    }
    return colors;
}

// ============================================================================
// Loc scenery: models, loc configs, and placements.
// ============================================================================

// Cursor with the client's variable-length integer reads (model and map
// streams lean on these heavily).
struct Cursor {
    const uint8_t* d;
    size_t n;
    size_t off = 0;

    explicit Cursor(const std::vector<uint8_t>& v) : d(v.data()), n(v.size()) {}

    void check(size_t k) const {
        if (off + k > n) throw std::runtime_error("scenery: buffer underflow");
    }
    uint8_t u8() { check(1); return d[off++]; }
    int8_t i8() { return static_cast<int8_t>(u8()); }
    uint16_t u16() { check(2); const uint16_t v = static_cast<uint16_t>((d[off] << 8) | d[off + 1]); off += 2; return v; }
    int16_t i16() { return static_cast<int16_t>(u16()); }
    uint32_t u24() { check(3); const uint32_t v = (uint32_t(d[off]) << 16) | (uint32_t(d[off + 1]) << 8) | d[off + 2]; off += 3; return v; }
    int32_t i32() { check(4); int32_t v = static_cast<int32_t>((uint32_t(d[off]) << 24) | (uint32_t(d[off + 1]) << 16) | (uint32_t(d[off + 2]) << 8) | d[off + 3]); off += 4; return v; }
    // signed "smart2": one byte biased by 64, or two bytes biased by 49152
    int32_t smart2() { check(1); return static_cast<int8_t>(d[off]) >= 0 ? int32_t(u8()) - 64 : int32_t(u16()) - 49152; }
    // unsigned smart: one byte < 128, else two bytes biased by 0x8000
    int32_t usmart() { check(1); return d[off] < 128 ? int32_t(u8()) : int32_t(u16()) - 0x8000; }
    // accumulating smart used for loc id deltas
    int32_t smart3() {
        int32_t total = 0;
        int32_t v = usmart();
        while (v == 32767) { total += 32767; v = usmart(); }
        return total + v;
    }
    void skipString() { while (u8() != 0) {} }
};

// Decoded subset of the classic (pre-2018) model format: geometry, face
// colors, render types, textures. Skins/priorities are walked but dropped -
// this viewer neither animates nor sorts.
struct RawModel {
    int32_t vertexCount = 0;
    int32_t faceCount = 0;
    std::vector<int32_t> vx, vy, vz;
    std::vector<int32_t> ia, ib, ic;
    std::vector<uint16_t> faceColors;
    std::vector<int8_t> faceRenderTypes; // empty = all smooth
    std::vector<int8_t> faceAlphas;      // empty = opaque
    std::vector<int16_t> faceTextures;   // empty = untextured
    // Texture-mapping triangles: three vertex indices whose positions define
    // the texture-space origin and axes for the faces that reference them
    // (via textureCoords); 0xffff marks an unsupported projection type.
    std::vector<int8_t> textureCoords;   // per face, -1 = use own vertices
    int32_t texTriangleCount = 0;
    std::vector<uint16_t> texP, texM, texN;
};

// The classic format is five parallel streams located via an 18-byte footer
// of section sizes; vertices are XYZ deltas gated by a flags stream, faces
// are a triangle fan/strip state machine over vertex-index deltas.
bool decodeClassicModel(const std::vector<uint8_t>& data, RawModel& out) {
    if (data.size() < 18) return false;
    // Newer footer-tagged formats (V1/V2/V3) are rare among world locs and
    // are skipped (counted by the caller).
    const int8_t tail1 = static_cast<int8_t>(data[data.size() - 1]);
    const int8_t tail2 = static_cast<int8_t>(data[data.size() - 2]);
    if (tail2 == -1 && (tail1 == -1 || tail1 == -2 || tail1 == -3)) return false;

    Cursor head(data);
    head.off = data.size() - 18;
    const int32_t vertexCount = head.u16();
    const int32_t faceCount = head.u16();
    const int32_t texTriangleCount = head.u8();
    const int32_t hasFaceInfo = head.u8();   // render type + texture flags
    const int32_t priorityByte = head.u8();  // 255 = per-face priorities
    const int32_t hasFaceAlphas = head.u8();
    const int32_t hasFaceSkins = head.u8();
    const int32_t hasVertexSkins = head.u8();
    const int32_t vertXSize = head.u16();
    const int32_t vertYSize = head.u16();
    head.u16(); // vertex Z stream size (implicit, trailing)
    const int32_t faceIndexSize = head.u16();

    size_t offset = 0;
    const size_t vertexFlagsOff = offset;      offset += static_cast<size_t>(vertexCount);
    const size_t faceTypesOff = offset;        offset += static_cast<size_t>(faceCount);
    const size_t prioritiesOff = offset;       if (priorityByte == 255) offset += static_cast<size_t>(faceCount);
    const size_t faceSkinsOff = offset;        if (hasFaceSkins == 1) offset += static_cast<size_t>(faceCount);
    const size_t faceInfoOff = offset;         if (hasFaceInfo == 1) offset += static_cast<size_t>(faceCount);
    const size_t vertexSkinsOff = offset;      if (hasVertexSkins == 1) offset += static_cast<size_t>(vertexCount);
    const size_t alphasOff = offset;           if (hasFaceAlphas == 1) offset += static_cast<size_t>(faceCount);
    const size_t faceIndicesOff = offset;      offset += static_cast<size_t>(faceIndexSize);
    const size_t colorsOff = offset;           offset += static_cast<size_t>(faceCount) * 2;
    const size_t texMappingOff = offset;       offset += static_cast<size_t>(texTriangleCount) * 6;
    const size_t vertXOff = offset;            offset += static_cast<size_t>(vertXSize);
    const size_t vertYOff = offset;            offset += static_cast<size_t>(vertYSize);
    const size_t vertZOff = offset;
    (void)prioritiesOff;
    (void)faceSkinsOff;
    (void)vertexSkinsOff;

    out.vertexCount = vertexCount;
    out.faceCount = faceCount;
    out.vx.resize(static_cast<size_t>(vertexCount));
    out.vy.resize(static_cast<size_t>(vertexCount));
    out.vz.resize(static_cast<size_t>(vertexCount));
    out.ia.resize(static_cast<size_t>(faceCount));
    out.ib.resize(static_cast<size_t>(faceCount));
    out.ic.resize(static_cast<size_t>(faceCount));
    out.faceColors.resize(static_cast<size_t>(faceCount));
    if (hasFaceInfo == 1) {
        out.faceRenderTypes.resize(static_cast<size_t>(faceCount));
        out.faceTextures.assign(static_cast<size_t>(faceCount), -1);
        out.textureCoords.assign(static_cast<size_t>(faceCount), -1);
    }
    if (hasFaceAlphas == 1) out.faceAlphas.resize(static_cast<size_t>(faceCount));

    // Vertices: per-vertex presence flags select which axes carry a delta.
    Cursor flags(data);   flags.off = vertexFlagsOff;
    Cursor xs(data);      xs.off = vertXOff;
    Cursor ys(data);      ys.off = vertYOff;
    Cursor zs(data);      zs.off = vertZOff;
    int32_t px = 0;
    int32_t py = 0;
    int32_t pz = 0;
    for (int32_t i = 0; i < vertexCount; ++i) {
        const uint8_t flag = flags.u8();
        px += (flag & 1) != 0 ? xs.smart2() : 0;
        py += (flag & 2) != 0 ? ys.smart2() : 0;
        pz += (flag & 4) != 0 ? zs.smart2() : 0;
        out.vx[static_cast<size_t>(i)] = px;
        out.vy[static_cast<size_t>(i)] = py;
        out.vz[static_cast<size_t>(i)] = pz;
    }

    // Face colors + optional render-type/texture flags + alphas.
    Cursor colors(data); colors.off = colorsOff;
    Cursor info(data);   info.off = faceInfoOff;
    Cursor alphas(data); alphas.off = alphasOff;
    for (int32_t i = 0; i < faceCount; ++i) {
        out.faceColors[static_cast<size_t>(i)] = colors.u16();
        if (hasFaceInfo == 1) {
            const uint8_t flag = info.u8();
            out.faceRenderTypes[static_cast<size_t>(i)] = (flag & 1) != 0 ? 1 : 0;
            if ((flag & 2) != 0) {
                // Textured face: the color slot actually held the texture id,
                // and the flag's upper bits name the texture-mapping triangle.
                out.faceTextures[static_cast<size_t>(i)] =
                    static_cast<int16_t>(out.faceColors[static_cast<size_t>(i)]);
                out.faceColors[static_cast<size_t>(i)] = 127;
                out.textureCoords[static_cast<size_t>(i)] = static_cast<int8_t>(flag >> 2);
            }
        }
        if (hasFaceAlphas == 1) out.faceAlphas[static_cast<size_t>(i)] = alphas.i8();
    }

    out.texTriangleCount = texTriangleCount;
    if (texTriangleCount > 0) {
        Cursor tm(data);
        tm.off = texMappingOff;
        out.texP.resize(static_cast<size_t>(texTriangleCount));
        out.texM.resize(static_cast<size_t>(texTriangleCount));
        out.texN.resize(static_cast<size_t>(texTriangleCount));
        for (int32_t i = 0; i < texTriangleCount; ++i) {
            out.texP[static_cast<size_t>(i)] = tm.u16();
            out.texM[static_cast<size_t>(i)] = tm.u16();
            out.texN[static_cast<size_t>(i)] = tm.u16();
        }
    }

    // Faces: strip/fan opcodes over delta-encoded vertex indices.
    Cursor faceIdx(data);   faceIdx.off = faceIndicesOff;
    Cursor faceTypes(data); faceTypes.off = faceTypesOff;
    int32_t a = 0;
    int32_t b = 0;
    int32_t c = 0;
    int32_t last = 0;
    for (int32_t i = 0; i < faceCount; ++i) {
        const uint8_t type = faceTypes.u8();
        switch (type) {
            case 1:
                a = faceIdx.smart2() + last;
                b = faceIdx.smart2() + a;
                c = faceIdx.smart2() + b;
                last = c;
                break;
            case 2:
                b = c;
                c = faceIdx.smart2() + last;
                last = c;
                break;
            case 3:
                a = c;
                c = faceIdx.smart2() + last;
                last = c;
                break;
            case 4: {
                const int32_t tmp = a;
                a = b;
                b = tmp;
                c = faceIdx.smart2() + last;
                last = c;
                break;
            }
            default:
                return false;
        }
        if (a < 0 || b < 0 || c < 0 || a >= vertexCount || b >= vertexCount ||
            c >= vertexCount) {
            return false;
        }
        out.ia[static_cast<size_t>(i)] = a;
        out.ib[static_cast<size_t>(i)] = b;
        out.ic[static_cast<size_t>(i)] = c;
    }
    return true;
}

// The V2 format (footer bytes -1,-2): same streams as the classic format,
// but a 23-byte footer with two extra section sizes, and the alpha/index
// streams shuffled to make room for a skeletal-animation blob we skip.
bool decodeV2Model(const std::vector<uint8_t>& data, RawModel& out) {
    if (data.size() < 23) return false;

    Cursor head(data);
    head.off = data.size() - 23;
    const int32_t vertexCount = head.u16();
    const int32_t faceCount = head.u16();
    const int32_t texTriangleCount = head.u8();
    const int32_t hasFaceInfo = head.u8();
    const int32_t priorityByte = head.u8();
    const int32_t hasFaceAlphas = head.u8();
    const int32_t hasFaceSkins = head.u8();
    head.u8(); // hasVertexSkins - inside the skins blob, skipped whole below
    head.u8(); // hasMayaGroups  - likewise
    const int32_t vertXSize = head.u16();
    const int32_t vertYSize = head.u16();
    head.u16(); // vertex Z stream size (implicit, trailing)
    const int32_t faceIndexSize = head.u16();
    const int32_t skinsSize = head.u16();

    size_t offset = 0;
    const size_t vertexFlagsOff = offset; offset += static_cast<size_t>(vertexCount);
    const size_t faceTypesOff = offset;   offset += static_cast<size_t>(faceCount);
    if (priorityByte == 255) offset += static_cast<size_t>(faceCount);
    if (hasFaceSkins == 1) offset += static_cast<size_t>(faceCount);
    const size_t faceInfoOff = offset;    if (hasFaceInfo == 1) offset += static_cast<size_t>(faceCount);
    offset += static_cast<size_t>(skinsSize); // vertex skins + maya groups
    const size_t alphasOff = offset;      if (hasFaceAlphas == 1) offset += static_cast<size_t>(faceCount);
    const size_t faceIndicesOff = offset; offset += static_cast<size_t>(faceIndexSize);
    const size_t colorsOff = offset;      offset += static_cast<size_t>(faceCount) * 2;
    const size_t texMappingOff = offset;  offset += static_cast<size_t>(texTriangleCount) * 6;
    const size_t vertXOff = offset;       offset += static_cast<size_t>(vertXSize);
    const size_t vertYOff = offset;       offset += static_cast<size_t>(vertYSize);
    const size_t vertZOff = offset;

    out.vertexCount = vertexCount;
    out.faceCount = faceCount;
    out.vx.resize(static_cast<size_t>(vertexCount));
    out.vy.resize(static_cast<size_t>(vertexCount));
    out.vz.resize(static_cast<size_t>(vertexCount));
    out.ia.resize(static_cast<size_t>(faceCount));
    out.ib.resize(static_cast<size_t>(faceCount));
    out.ic.resize(static_cast<size_t>(faceCount));
    out.faceColors.resize(static_cast<size_t>(faceCount));
    if (hasFaceInfo == 1) {
        out.faceRenderTypes.resize(static_cast<size_t>(faceCount));
        out.faceTextures.assign(static_cast<size_t>(faceCount), -1);
        out.textureCoords.assign(static_cast<size_t>(faceCount), -1);
    }
    if (hasFaceAlphas == 1) out.faceAlphas.resize(static_cast<size_t>(faceCount));
    out.texTriangleCount = texTriangleCount;
    if (texTriangleCount > 0) {
        Cursor tm(data);
        tm.off = texMappingOff;
        out.texP.resize(static_cast<size_t>(texTriangleCount));
        out.texM.resize(static_cast<size_t>(texTriangleCount));
        out.texN.resize(static_cast<size_t>(texTriangleCount));
        for (int32_t i = 0; i < texTriangleCount; ++i) {
            out.texP[static_cast<size_t>(i)] = tm.u16();
            out.texM[static_cast<size_t>(i)] = tm.u16();
            out.texN[static_cast<size_t>(i)] = tm.u16();
        }
    }

    Cursor flags(data); flags.off = vertexFlagsOff;
    Cursor xs(data);    xs.off = vertXOff;
    Cursor ys(data);    ys.off = vertYOff;
    Cursor zs(data);    zs.off = vertZOff;
    int32_t px = 0;
    int32_t py = 0;
    int32_t pz = 0;
    for (int32_t i = 0; i < vertexCount; ++i) {
        const uint8_t flag = flags.u8();
        px += (flag & 1) != 0 ? xs.smart2() : 0;
        py += (flag & 2) != 0 ? ys.smart2() : 0;
        pz += (flag & 4) != 0 ? zs.smart2() : 0;
        out.vx[static_cast<size_t>(i)] = px;
        out.vy[static_cast<size_t>(i)] = py;
        out.vz[static_cast<size_t>(i)] = pz;
    }

    Cursor colors(data); colors.off = colorsOff;
    Cursor info(data);   info.off = faceInfoOff;
    Cursor alphas(data); alphas.off = alphasOff;
    for (int32_t i = 0; i < faceCount; ++i) {
        out.faceColors[static_cast<size_t>(i)] = colors.u16();
        if (hasFaceInfo == 1) {
            const uint8_t flag = info.u8();
            out.faceRenderTypes[static_cast<size_t>(i)] = (flag & 1) != 0 ? 1 : 0;
            if ((flag & 2) != 0) {
                out.faceTextures[static_cast<size_t>(i)] =
                    static_cast<int16_t>(out.faceColors[static_cast<size_t>(i)]);
                out.faceColors[static_cast<size_t>(i)] = 127;
                out.textureCoords[static_cast<size_t>(i)] = static_cast<int8_t>(flag >> 2);
            }
        }
        if (hasFaceAlphas == 1) out.faceAlphas[static_cast<size_t>(i)] = alphas.i8();
    }

    Cursor faceIdx(data);   faceIdx.off = faceIndicesOff;
    Cursor faceTypes(data); faceTypes.off = faceTypesOff;
    int32_t a = 0;
    int32_t b = 0;
    int32_t c = 0;
    int32_t last = 0;
    for (int32_t i = 0; i < faceCount; ++i) {
        const uint8_t type = faceTypes.u8();
        switch (type) {
            case 1:
                a = faceIdx.smart2() + last;
                b = faceIdx.smart2() + a;
                c = faceIdx.smart2() + b;
                last = c;
                break;
            case 2: b = c; c = faceIdx.smart2() + last; last = c; break;
            case 3: a = c; c = faceIdx.smart2() + last; last = c; break;
            case 4: {
                const int32_t tmp = a;
                a = b;
                b = tmp;
                c = faceIdx.smart2() + last;
                last = c;
                break;
            }
            default: return false;
        }
        if (a < 0 || b < 0 || c < 0 || a >= vertexCount || b >= vertexCount ||
            c >= vertexCount) {
            return false;
        }
        out.ia[static_cast<size_t>(i)] = a;
        out.ib[static_cast<size_t>(i)] = b;
        out.ic[static_cast<size_t>(i)] = c;
    }
    return true;
}

// The V3 format (footer bytes -1,-3): the "new" model layout. Texture
// render types lead the file, face render types get their own byte stream
// (no longer packed into texture flags), face textures are u16-1, and
// complex texture projections trail the file (walked, unused here).
bool decodeV3Model(const std::vector<uint8_t>& data, RawModel& out) {
    if (data.size() < 26) return false;

    Cursor head(data);
    head.off = data.size() - 26;
    const int32_t vertexCount = head.u16();
    const int32_t faceCount = head.u16();
    const int32_t texTriangleCount = head.u8();
    const int32_t hasFaceInfo = head.u8();
    const int32_t priorityByte = head.u8();
    const int32_t hasFaceAlphas = head.u8();
    const int32_t hasFaceSkins = head.u8();
    const int32_t hasFaceTextures = head.u8();
    head.u8(); // hasVertexSkins - inside the skins blob, skipped whole
    head.u8(); // hasMayaGroups  - likewise
    const int32_t vertXSize = head.u16();
    const int32_t vertYSize = head.u16();
    const int32_t vertZSize = head.u16();
    const int32_t faceIndexSize = head.u16();
    const int32_t texCoordsSize = head.u16();
    const int32_t skinsSize = head.u16();

    // Texture render types prefix the file; only "simple" (type 0) entries
    // occupy the flat P/M/N mapping stream (types 1-3 are projected
    // cylinder/cube mappings this viewer approximates as untextured coords).
    if (static_cast<size_t>(texTriangleCount) > data.size()) return false;
    std::vector<int8_t> texRenderTypes(static_cast<size_t>(texTriangleCount));
    {
        Cursor tt(data);
        for (int32_t i = 0; i < texTriangleCount; ++i) {
            texRenderTypes[static_cast<size_t>(i)] = tt.i8();
        }
    }

    size_t offset = static_cast<size_t>(texTriangleCount) + static_cast<size_t>(vertexCount);
    const size_t renderTypesOff = offset;  if (hasFaceInfo == 1) offset += static_cast<size_t>(faceCount);
    const size_t faceTypesOff = offset;    offset += static_cast<size_t>(faceCount);
    if (priorityByte == 255) offset += static_cast<size_t>(faceCount);
    if (hasFaceSkins == 1) offset += static_cast<size_t>(faceCount);
    offset += static_cast<size_t>(skinsSize);
    const size_t alphasOff = offset;       if (hasFaceAlphas == 1) offset += static_cast<size_t>(faceCount);
    const size_t faceIndicesOff = offset;  offset += static_cast<size_t>(faceIndexSize);
    const size_t texturesOff = offset;     if (hasFaceTextures == 1) offset += static_cast<size_t>(faceCount) * 2;
    const size_t texCoordsOff = offset;    offset += static_cast<size_t>(texCoordsSize);
    const size_t colorsOff = offset;       offset += static_cast<size_t>(faceCount) * 2;
    const size_t vertXOff = offset;        offset += static_cast<size_t>(vertXSize);
    const size_t vertYOff = offset;        offset += static_cast<size_t>(vertYSize);
    const size_t vertZOff = offset;
    const size_t texMappingOff = vertZOff + static_cast<size_t>(vertZSize);

    out.vertexCount = vertexCount;
    out.faceCount = faceCount;
    out.vx.resize(static_cast<size_t>(vertexCount));
    out.vy.resize(static_cast<size_t>(vertexCount));
    out.vz.resize(static_cast<size_t>(vertexCount));
    out.ia.resize(static_cast<size_t>(faceCount));
    out.ib.resize(static_cast<size_t>(faceCount));
    out.ic.resize(static_cast<size_t>(faceCount));
    out.faceColors.resize(static_cast<size_t>(faceCount));
    if (hasFaceInfo == 1) out.faceRenderTypes.resize(static_cast<size_t>(faceCount));
    if (hasFaceAlphas == 1) out.faceAlphas.resize(static_cast<size_t>(faceCount));
    if (hasFaceTextures == 1) {
        out.faceTextures.assign(static_cast<size_t>(faceCount), -1);
        out.textureCoords.assign(static_cast<size_t>(faceCount), -1);
    }
    out.texTriangleCount = texTriangleCount;
    if (texTriangleCount > 0) {
        Cursor tm(data);
        tm.off = texMappingOff;
        out.texP.assign(static_cast<size_t>(texTriangleCount), 0xffff);
        out.texM.assign(static_cast<size_t>(texTriangleCount), 0xffff);
        out.texN.assign(static_cast<size_t>(texTriangleCount), 0xffff);
        for (int32_t i = 0; i < texTriangleCount; ++i) {
            if (texRenderTypes[static_cast<size_t>(i)] == 0) {
                out.texP[static_cast<size_t>(i)] = tm.u16();
                out.texM[static_cast<size_t>(i)] = tm.u16();
                out.texN[static_cast<size_t>(i)] = tm.u16();
            }
        }
    }

    Cursor flags(data); flags.off = static_cast<size_t>(texTriangleCount);
    Cursor xs(data);    xs.off = vertXOff;
    Cursor ys(data);    ys.off = vertYOff;
    Cursor zs(data);    zs.off = vertZOff;
    int32_t px = 0;
    int32_t py = 0;
    int32_t pz = 0;
    for (int32_t i = 0; i < vertexCount; ++i) {
        const uint8_t flag = flags.u8();
        px += (flag & 1) != 0 ? xs.smart2() : 0;
        py += (flag & 2) != 0 ? ys.smart2() : 0;
        pz += (flag & 4) != 0 ? zs.smart2() : 0;
        out.vx[static_cast<size_t>(i)] = px;
        out.vy[static_cast<size_t>(i)] = py;
        out.vz[static_cast<size_t>(i)] = pz;
    }

    Cursor colors(data);   colors.off = colorsOff;
    Cursor types(data);    types.off = renderTypesOff;
    Cursor alphas(data);   alphas.off = alphasOff;
    Cursor textures(data); textures.off = texturesOff;
    Cursor coords(data);   coords.off = texCoordsOff;
    for (int32_t i = 0; i < faceCount; ++i) {
        out.faceColors[static_cast<size_t>(i)] = colors.u16();
        if (hasFaceInfo == 1) out.faceRenderTypes[static_cast<size_t>(i)] = types.i8();
        if (hasFaceAlphas == 1) out.faceAlphas[static_cast<size_t>(i)] = alphas.i8();
        if (hasFaceTextures == 1) {
            out.faceTextures[static_cast<size_t>(i)] =
                static_cast<int16_t>(static_cast<int32_t>(textures.u16()) - 1);
            // The coord stream only has entries for textured faces, and only
            // when texture triangles exist at all.
            if (out.faceTextures[static_cast<size_t>(i)] != -1 && texCoordsSize > 0) {
                out.textureCoords[static_cast<size_t>(i)] =
                    static_cast<int8_t>(static_cast<int32_t>(coords.u8()) - 1);
            }
        }
    }

    Cursor faceIdx(data);   faceIdx.off = faceIndicesOff;
    Cursor faceTypes(data); faceTypes.off = faceTypesOff;
    int32_t a = 0;
    int32_t b = 0;
    int32_t c = 0;
    int32_t last = 0;
    for (int32_t i = 0; i < faceCount; ++i) {
        const uint8_t type = faceTypes.u8();
        switch (type) {
            case 1:
                a = faceIdx.smart2() + last;
                b = faceIdx.smart2() + a;
                c = faceIdx.smart2() + b;
                last = c;
                break;
            case 2: b = c; c = faceIdx.smart2() + last; last = c; break;
            case 3: a = c; c = faceIdx.smart2() + last; last = c; break;
            case 4: {
                const int32_t tmp = a;
                a = b;
                b = tmp;
                c = faceIdx.smart2() + last;
                last = c;
                break;
            }
            default: return false;
        }
        if (a < 0 || b < 0 || c < 0 || a >= vertexCount || b >= vertexCount ||
            c >= vertexCount) {
            return false;
        }
        out.ia[static_cast<size_t>(i)] = a;
        out.ib[static_cast<size_t>(i)] = b;
        out.ic[static_cast<size_t>(i)] = c;
    }
    return true;
}

// Dispatch on the footer tag. Only V1 (-1,-1) remains unported (absent from
// the caches this targets); the caller counts it as skipped.
bool decodeModel(const std::vector<uint8_t>& data, RawModel& out) {
    if (data.size() < 2) return false;
    const int8_t tail1 = static_cast<int8_t>(data[data.size() - 1]);
    const int8_t tail2 = static_cast<int8_t>(data[data.size() - 2]);
    if (tail2 == -1 && tail1 == -3) return decodeV3Model(data, out);
    if (tail2 == -1 && tail1 == -2) return decodeV2Model(data, out);
    if (tail2 == -1 && tail1 == -1) return false;
    return decodeClassicModel(data, out);
}

// The subset of a loc config the viewer needs: which models to show for
// which shape, footprint, recolors, scale/offset, lighting tweaks.
struct LocType {
    std::vector<std::pair<int32_t, std::vector<int32_t>>> shapeModels; // (shape, models); shape -1 = any
    int32_t sizeX = 1;
    int32_t sizeY = 1;
    int32_t ambient = 0;
    int32_t contrast = 0;
    std::vector<std::pair<uint16_t, uint16_t>> recolor;
    std::vector<std::pair<uint16_t, uint16_t>> retexture;
    int32_t modelSizeX = 128;
    int32_t modelSizeHeight = 128;
    int32_t modelSizeY = 128;
    int32_t offsetX = 0;
    int32_t offsetHeight = 0;
    int32_t offsetY = 0;
};

// Opcode walk for OSRS rev 220+ loc configs. Fields the viewer ignores must
// still be consumed exactly, or every later field in the file misparses.
LocType decodeLocType(const std::vector<uint8_t>& data) {
    LocType type;
    Cursor r(data);
    while (r.off < r.n) {
        const uint8_t opcode = r.u8();
        if (opcode == 0) break;
        switch (opcode) {
            case 1: {
                const int count = r.u8();
                for (int i = 0; i < count; ++i) {
                    const int32_t modelId = r.u16();
                    const int32_t shape = r.u8();
                    type.shapeModels.push_back({shape, {modelId}});
                }
                break;
            }
            case 2:
            case 3: r.skipString(); break;
            case 5: {
                const int count = r.u8();
                std::vector<int32_t> models;
                for (int i = 0; i < count; ++i) models.push_back(r.u16());
                if (count > 0) type.shapeModels.push_back({-1, std::move(models)});
                break;
            }
            // Recent OSRS revisions outgrew 16-bit model ids; opcodes 6/7
            // are the wide versions of 1/5 (32-bit model ids).
            case 6: {
                const int count = r.u8();
                for (int i = 0; i < count; ++i) {
                    const int32_t modelId = r.i32();
                    const int32_t shape = r.u8();
                    type.shapeModels.push_back({shape, {modelId}});
                }
                break;
            }
            case 7: {
                const int count = r.u8();
                std::vector<int32_t> models;
                for (int i = 0; i < count; ++i) models.push_back(r.i32());
                if (count > 0) type.shapeModels.push_back({-1, std::move(models)});
                break;
            }
            case 14: type.sizeX = r.u8(); break;
            case 15: type.sizeY = r.u8(); break;
            case 17: case 18: break;
            case 19: r.u8(); break;
            case 21: case 22: case 23: break;
            case 24: r.u16(); break;
            case 25: case 27: break;
            case 28: r.u8(); break;
            case 29: type.ambient = r.i8(); break;
            case 39: type.contrast = r.i8() * 25; break;
            case 30: case 31: case 32: case 33: case 34:
            case 35: case 36: case 37: case 38: r.skipString(); break;
            case 40: {
                const int count = r.u8();
                for (int i = 0; i < count; ++i) {
                    const uint16_t from = r.u16();
                    const uint16_t to = r.u16();
                    type.recolor.push_back({from, to});
                }
                break;
            }
            case 41: {
                const int count = r.u8();
                for (int i = 0; i < count; ++i) {
                    const uint16_t from = r.u16();
                    const uint16_t to = r.u16();
                    type.retexture.push_back({from, to});
                }
                break;
            }
            case 44: case 45: r.u16(); break;
            case 60: case 61: r.u16(); break;
            case 62: case 64: break;
            case 65: type.modelSizeX = r.u16(); break;
            case 66: type.modelSizeHeight = r.u16(); break;
            case 67: type.modelSizeY = r.u16(); break;
            case 68: r.u16(); break;
            case 69: r.u8(); break;
            case 70: type.offsetX = r.i16(); break;
            case 71: type.offsetHeight = r.i16(); break;
            case 72: type.offsetY = r.i16(); break;
            case 73: case 74: break;
            case 75: r.u8(); break;
            case 77:
            case 92: {
                r.u16(); // varbit
                r.u16(); // varp
                if (opcode == 92) r.u16();
                const int count = r.u8();
                for (int i = 0; i <= count; ++i) r.u16();
                break;
            }
            case 78: r.u16(); r.u8(); r.u8(); break;
            case 79: {
                r.u16(); r.u16(); r.u8(); r.u8();
                const int count = r.u8();
                for (int i = 0; i < count; ++i) r.u16();
                break;
            }
            case 80: {
                const int count = r.u8();
                for (int i = 0; i < count; ++i) { r.u16(); r.u16(); }
                break;
            }
            case 81: r.u8(); break;
            case 82: r.u16(); break;
            case 88: case 89: case 90: break;
            case 91: r.u8(); break;
            case 93: r.u8(); r.u16(); r.u8(); r.u16(); break;
            case 94: break;
            case 95: r.u8(); break;
            case 249: {
                const int count = r.u8();
                for (int i = 0; i < count; ++i) {
                    const bool isString = r.u8() == 1;
                    r.u24();
                    if (isString) r.skipString();
                    else r.i32();
                }
                break;
            }
            default:
                // Unknown opcode: cannot resync, keep what we have.
                return type;
        }
    }
    return type;
}

// One placement from a map square's loc file.
struct LocPlacement {
    int32_t id;
    int32_t shape;
    int32_t rotation;
    int32_t level;
    int32_t x; // scene tiles
    int32_t y;
};

void decodeLocPlacements(const std::vector<uint8_t>& data, int offsetX, int offsetY,
                         std::vector<LocPlacement>& out) {
    Cursor r(data);
    int32_t id = -1;
    while (true) {
        const int32_t idDelta = r.smart3();
        if (idDelta == 0) break;
        id += idDelta;
        int32_t pos = 0;
        while (true) {
            const int32_t posDelta = r.usmart();
            if (posDelta == 0) break;
            pos += posDelta - 1;
            const int32_t localY = pos & 0x3f;
            const int32_t localX = (pos >> 6) & 0x3f;
            const int32_t level = pos >> 12;
            const uint8_t attributes = r.u8();
            out.push_back({id, attributes >> 2, attributes & 3, level,
                           localX + offsetX, localY + offsetY});
        }
    }
}

// What the scene builder knows about one game texture.
struct TextureInfo {
    int32_t avgHsl = 0;  // packed HSL average, the minimap substitute
    int32_t layer = -1;  // layer in the uploaded texture array, -1 = none
};

// Decodes a cache sprite sheet (index 8) and rasterizes its first frame
// into a 128x128 RGBA image. Sprites are palettized with palette entry 0
// meaning "transparent" - which is exactly what leaf/roof cutouts rely on.
// Pixels can be stored row- or column-major, and the sub-image sits inside
// a canvas at an offset.
bool decodeSpriteRgba(const std::vector<uint8_t>& data, double brightness,
                      std::vector<uint8_t>& outRgba) {
    if (data.size() < 9) return false;
    Cursor tail(data);
    tail.off = data.size() - 2;
    const int32_t spriteCount = tail.u16();
    if (spriteCount < 1) return false;

    const size_t headerOff = data.size() - 7 - static_cast<size_t>(spriteCount) * 8;
    Cursor head(data);
    head.off = headerOff;
    const int32_t canvasW = head.u16();
    const int32_t canvasH = head.u16();
    const int32_t paletteSize = head.u8() + 1;
    // The per-sprite metadata is stored field-major (all x offsets, then all
    // y offsets, then widths, then heights); only the sheet's first sprite
    // is used for a texture.
    const size_t fieldsOff = headerOff + 5;
    auto fieldAt = [&](size_t fieldIndex) {
        Cursor c(data);
        c.off = fieldsOff + fieldIndex * static_cast<size_t>(spriteCount) * 2;
        return static_cast<int32_t>(c.u16());
    };
    const int32_t xOff = fieldAt(0);
    const int32_t yOff = fieldAt(1);
    const int32_t width = fieldAt(2);
    const int32_t height = fieldAt(3);
    if (canvasW <= 0 || canvasH <= 0 || width <= 0 || height <= 0) return false;

    Cursor pal(data);
    pal.off = headerOff - static_cast<size_t>(paletteSize - 1) * 3;
    std::vector<int32_t> palette(static_cast<size_t>(paletteSize), 0);
    for (int32_t i = 1; i < paletteSize; ++i) {
        palette[static_cast<size_t>(i)] = static_cast<int32_t>(pal.u24());
        if (palette[static_cast<size_t>(i)] == 0) palette[static_cast<size_t>(i)] = 1;
        palette[static_cast<size_t>(i)] =
            brightenRgb(palette[static_cast<size_t>(i)], brightness);
    }

    Cursor px(data);
    const uint8_t flags = px.u8();
    const bool columnWise = (flags & 1) != 0;
    const size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
    std::vector<uint8_t> indices(pixelCount, 0);
    if (!columnWise) {
        for (size_t i = 0; i < pixelCount; ++i) indices[i] = px.u8();
    } else {
        for (int32_t x = 0; x < width; ++x) {
            for (int32_t y = 0; y < height; ++y) {
                indices[static_cast<size_t>(x) + static_cast<size_t>(y) * width] = px.u8();
            }
        }
    }

    // Compose onto the canvas, then nearest-sample the canvas to 128x128.
    outRgba.assign(128 * 128 * 4, 0);
    for (int32_t oy = 0; oy < 128; ++oy) {
        for (int32_t ox = 0; ox < 128; ++ox) {
            const int32_t cx = ox * canvasW / 128;
            const int32_t cy = oy * canvasH / 128;
            const int32_t sx = cx - xOff;
            const int32_t sy = cy - yOff;
            uint8_t index = 0;
            if (sx >= 0 && sy >= 0 && sx < width && sy < height) {
                index = indices[static_cast<size_t>(sx) +
                                static_cast<size_t>(sy) * width];
            }
            uint8_t* out = outRgba.data() + (static_cast<size_t>(oy) * 128 + ox) * 4;
            if (index != 0 && index < paletteSize) {
                const int32_t rgb = palette[index];
                out[0] = static_cast<uint8_t>((rgb >> 16) & 0xff);
                out[1] = static_cast<uint8_t>((rgb >> 8) & 0xff);
                out[2] = static_cast<uint8_t>(rgb & 0xff);
                out[3] = 0xff;
            } // else: transparent cutout (all zero)
        }
    }
    return true;
}

// A lit model ready for stamping into the scene: triangle soup in model
// units with per-vertex packed HSL already containing the light term.
// Textured faces carry UVs, an array layer, and light-only "HSL" values
// (the texture supplies the color; the vertex carries the shade).
struct LitModel {
    std::vector<glm::ivec3> positions; // 3 per face
    std::vector<int32_t> hsls;         // 3 per face
    std::vector<glm::vec2> uvs;        // 3 per face
    std::vector<int32_t> layers;       // 1 per face, -1 = untextured
};

// UVs for one textured face. The model format expresses texture space as a
// triangle of vertex positions (P = origin, M = u axis, N = v axis); each
// face vertex is decomposed into that basis. When no mapping triangle is
// referenced, the face's own corners span the texture.
void computeFaceUvs(const RawModel& raw, int32_t face, glm::vec2 outUv[3]) {
    int32_t p = raw.ia[static_cast<size_t>(face)];
    int32_t m = raw.ib[static_cast<size_t>(face)];
    int32_t n = raw.ic[static_cast<size_t>(face)];
    const int32_t coordRaw =
        raw.textureCoords.empty() ? -1 : raw.textureCoords[static_cast<size_t>(face)];
    if (coordRaw != -1) {
        const int32_t coord = coordRaw & 0xff;
        if (coord < raw.texTriangleCount &&
            raw.texP[static_cast<size_t>(coord)] != 0xffff) {
            p = raw.texP[static_cast<size_t>(coord)];
            m = raw.texM[static_cast<size_t>(coord)];
            n = raw.texN[static_cast<size_t>(coord)];
        }
    }
    if (p >= raw.vertexCount || m >= raw.vertexCount || n >= raw.vertexCount) {
        outUv[0] = {0.0f, 0.0f};
        outUv[1] = {1.0f, 0.0f};
        outUv[2] = {0.0f, 1.0f};
        return;
    }

    const glm::dvec3 origin(raw.vx[static_cast<size_t>(p)], raw.vy[static_cast<size_t>(p)],
                            raw.vz[static_cast<size_t>(p)]);
    const glm::dvec3 uAxis =
        glm::dvec3(raw.vx[static_cast<size_t>(m)], raw.vy[static_cast<size_t>(m)],
                   raw.vz[static_cast<size_t>(m)]) - origin;
    const glm::dvec3 vAxis =
        glm::dvec3(raw.vx[static_cast<size_t>(n)], raw.vy[static_cast<size_t>(n)],
                   raw.vz[static_cast<size_t>(n)]) - origin;
    const glm::dvec3 normal = glm::cross(uAxis, vAxis);
    const double denom = glm::dot(normal, normal);
    if (denom < 1e-9) {
        outUv[0] = {0.0f, 0.0f};
        outUv[1] = {1.0f, 0.0f};
        outUv[2] = {0.0f, 1.0f};
        return;
    }

    const int32_t corners[3] = {raw.ia[static_cast<size_t>(face)],
                                raw.ib[static_cast<size_t>(face)],
                                raw.ic[static_cast<size_t>(face)]};
    for (int k = 0; k < 3; ++k) {
        const glm::dvec3 w =
            glm::dvec3(raw.vx[static_cast<size_t>(corners[k])],
                       raw.vy[static_cast<size_t>(corners[k])],
                       raw.vz[static_cast<size_t>(corners[k])]) - origin;
        outUv[k] = glm::vec2(
            static_cast<float>(glm::dot(glm::cross(w, vAxis), normal) / denom),
            static_cast<float>(glm::dot(glm::cross(uAxis, w), normal) / denom));
    }
}

// The client's model lighting (the actual "Gouraud" of this whole project):
// vertex normals accumulate face normals of smooth faces; each vertex color
// is the face's palette HSL with its lightness scaled by ambient + N.L.
// Flat-shaded faces (render type 1) use the face normal for all corners.
void lightModel(const RawModel& raw, int32_t ambient, int32_t contrast,
                const std::unordered_map<int32_t, TextureInfo>& textureInfo,
                LitModel& out) {
    const int32_t lightX = -50;
    const int32_t lightY = -10;
    const int32_t lightZ = -50;
    const int32_t magnitude = static_cast<int32_t>(
        std::sqrt(static_cast<double>(lightX * lightX + lightY * lightY + lightZ * lightZ)));
    const int32_t lightIntensity = (magnitude * contrast) >> 8;

    struct Normal { int32_t x = 0, y = 0, z = 0, count = 0; };
    std::vector<Normal> vertexNormals(static_cast<size_t>(raw.vertexCount));
    std::vector<Normal> faceNormals(static_cast<size_t>(raw.faceCount));

    for (int32_t i = 0; i < raw.faceCount; ++i) {
        const size_t fi = static_cast<size_t>(i);
        const int32_t a = raw.ia[fi];
        const int32_t b = raw.ib[fi];
        const int32_t c = raw.ic[fi];
        int32_t ux = raw.vx[static_cast<size_t>(b)] - raw.vx[static_cast<size_t>(a)];
        int32_t uy = raw.vy[static_cast<size_t>(b)] - raw.vy[static_cast<size_t>(a)];
        int32_t uz = raw.vz[static_cast<size_t>(b)] - raw.vz[static_cast<size_t>(a)];
        const int32_t wx = raw.vx[static_cast<size_t>(c)] - raw.vx[static_cast<size_t>(a)];
        const int32_t wy = raw.vy[static_cast<size_t>(c)] - raw.vy[static_cast<size_t>(a)];
        const int32_t wz = raw.vz[static_cast<size_t>(c)] - raw.vz[static_cast<size_t>(a)];
        int32_t nx = uy * wz - wy * uz;
        int32_t ny = uz * wx - wz * ux;
        int32_t nz = ux * wy - wx * uy;
        while (nx > 8192 || ny > 8192 || nz > 8192 || nx < -8192 || ny < -8192 ||
               nz < -8192) {
            nx >>= 1;
            ny >>= 1;
            nz >>= 1;
        }
        int32_t length = static_cast<int32_t>(std::sqrt(
            static_cast<double>(nx) * nx + static_cast<double>(ny) * ny +
            static_cast<double>(nz) * nz));
        if (length <= 0) length = 1;
        nx = nx * 256 / length;
        ny = ny * 256 / length;
        nz = nz * 256 / length;

        const int8_t type = raw.faceRenderTypes.empty() ? int8_t{0} : raw.faceRenderTypes[fi];
        if (type == 0) {
            for (const int32_t v : {a, b, c}) {
                Normal& normal = vertexNormals[static_cast<size_t>(v)];
                normal.x += nx;
                normal.y += ny;
                normal.z += nz;
                normal.count++;
            }
        } else {
            faceNormals[fi] = {nx, ny, nz, 1};
        }
        (void)uy;
    }

    auto vertexLight = [&](int32_t v) {
        const Normal& normal = vertexNormals[static_cast<size_t>(v)];
        const int32_t denom = lightIntensity * std::max(normal.count, 1);
        return ambient + (lightY * normal.y + lightZ * normal.z + lightX * normal.x) / denom;
    };

    for (int32_t i = 0; i < raw.faceCount; ++i) {
        const size_t fi = static_cast<size_t>(i);
        const int8_t alpha = raw.faceAlphas.empty() ? int8_t{0} : raw.faceAlphas[fi];
        int32_t type = raw.faceRenderTypes.empty() ? 0 : raw.faceRenderTypes[fi];
        if (alpha == -1) continue; // fully transparent
        if (alpha == -2) type = 3;

        const int32_t texture = raw.faceTextures.empty() ? -1 : raw.faceTextures[fi];
        const int32_t color = raw.faceColors[fi] & 0xffff;
        const int32_t a = raw.ia[fi];
        const int32_t b = raw.ib[fi];
        const int32_t c = raw.ic[fi];

        int32_t hslA;
        int32_t hslB;
        int32_t hslC;
        int32_t layer = -1;
        glm::vec2 uv[3] = {{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}};
        if (texture != -1) {
            auto it = textureInfo.find(texture);
            const TextureInfo info = it != textureInfo.end() ? it->second : TextureInfo{};
            // Per-vertex lightness; the texture supplies the color at
            // sampling time (or its average color if it failed to load).
            int32_t lightA;
            int32_t lightB;
            int32_t lightC;
            if (type == 0) {
                lightA = std::clamp(vertexLight(a), 2, 126);
                lightB = std::clamp(vertexLight(b), 2, 126);
                lightC = std::clamp(vertexLight(c), 2, 126);
            } else {
                const Normal& normal = faceNormals[fi];
                const int32_t light = std::clamp(
                    ambient + (lightY * normal.y + lightZ * normal.z + lightX * normal.x) /
                                  ((lightIntensity >> 1) + lightIntensity),
                    2, 126);
                lightA = lightB = lightC = light;
            }
            if (info.layer >= 0) {
                layer = info.layer;
                computeFaceUvs(raw, i, uv);
                hslA = lightA;
                hslB = lightB;
                hslC = lightC;
            } else {
                hslA = adjustUnderlayLight(info.avgHsl, lightA);
                hslB = adjustUnderlayLight(info.avgHsl, lightB);
                hslC = adjustUnderlayLight(info.avgHsl, lightC);
            }
        } else if (type == 0) {
            hslA = adjustUnderlayLight(color, vertexLight(a));
            hslB = adjustUnderlayLight(color, vertexLight(b));
            hslC = adjustUnderlayLight(color, vertexLight(c));
        } else if (type == 1) {
            const Normal& normal = faceNormals[fi];
            const int32_t light =
                ambient + (lightY * normal.y + lightZ * normal.z + lightX * normal.x) /
                              ((lightIntensity >> 1) + lightIntensity);
            hslA = hslB = hslC = adjustUnderlayLight(color, light);
        } else if (type == 3) {
            hslA = hslB = hslC = 128; // solid black
        } else {
            continue; // type 2: hidden
        }

        out.positions.push_back({raw.vx[static_cast<size_t>(a)], raw.vy[static_cast<size_t>(a)], raw.vz[static_cast<size_t>(a)]});
        out.positions.push_back({raw.vx[static_cast<size_t>(b)], raw.vy[static_cast<size_t>(b)], raw.vz[static_cast<size_t>(b)]});
        out.positions.push_back({raw.vx[static_cast<size_t>(c)], raw.vy[static_cast<size_t>(c)], raw.vz[static_cast<size_t>(c)]});
        out.hsls.push_back(hslA);
        out.hsls.push_back(hslB);
        out.hsls.push_back(hslC);
        out.uvs.push_back(uv[0]);
        out.uvs.push_back(uv[1]);
        out.uvs.push_back(uv[2]);
        out.layers.push_back(layer);
    }
}

float srgbToLinear(int32_t channel) {
    const float c = static_cast<float>(channel) / 255.0f;
    return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

// Emits one lit, palette-resolved triangle into the model (viewer space:
// 1 unit = 1 tile, +Y up, -Z north). Terrain colors arrive fully lit, so
// vertices carry a flat up normal and the renderer draws them unlit.
void emitTriangle(Model& model, const std::vector<int32_t>& palette,
                  const glm::vec3& a, const glm::vec3& b, const glm::vec3& c,
                  int32_t hslA, int32_t hslB, int32_t hslC, bool reorientUp,
                  const glm::vec2* uvs = nullptr, int32_t layer = -1) {
    // Untextured: the packed HSL resolves through the game palette.
    // Textured: the "hsl" is a bare lightness (2..126); the texture supplies
    // color at sampling time, so the vertex color becomes a gray multiplier.
    auto toColor = [&](int32_t hsl) {
        if (layer >= 0) {
            const int32_t gray = std::min(255, hsl * 255 / 128);
            return glm::vec3(srgbToLinear(gray));
        }
        const int32_t rgb = palette[static_cast<size_t>(std::clamp(hsl, 0, 65535))];
        return glm::vec3(srgbToLinear((rgb >> 16) & 0xff), srgbToLinear((rgb >> 8) & 0xff),
                         srgbToLinear(rgb & 0xff));
    };

    glm::vec3 v0 = a;
    glm::vec3 v1 = b;
    glm::vec3 v2 = c;
    glm::vec3 col0 = toColor(hslA);
    glm::vec3 col1 = toColor(hslB);
    glm::vec3 col2 = toColor(hslC);
    glm::vec3 uv0(0.0f, 0.0f, -1.0f);
    glm::vec3 uv1 = uv0;
    glm::vec3 uv2 = uv0;
    if (layer >= 0 && uvs != nullptr) {
        uv0 = glm::vec3(uvs[0], static_cast<float>(layer));
        uv1 = glm::vec3(uvs[1], static_cast<float>(layer));
        uv2 = glm::vec3(uvs[2], static_cast<float>(layer));
    }

    // Terrain winding is re-oriented upward (a height field's faces all
    // point up-ish); loc models keep their authored winding, which the
    // coordinate mapping preserves (its determinant is +1).
    const glm::vec3 normal = glm::cross(v1 - v0, v2 - v0);
    if (reorientUp && normal.y < 0.0f) {
        std::swap(v1, v2);
        std::swap(col1, col2);
        std::swap(uv1, uv2);
    }

    const glm::vec3 up(0.0f, 1.0f, 0.0f);
    const auto base = static_cast<uint32_t>(model.vertices.size());
    model.vertices.push_back({v0, up, col0, uv0});
    model.vertices.push_back({v1, up, col1, uv1});
    model.vertices.push_back({v2, up, col2, uv2});
    model.indices.push_back(base);
    model.indices.push_back(base + 1);
    model.indices.push_back(base + 2);
}

} // namespace

// ============================================================================
// WorldStreamer: square-independent state + per-square chunk building.
// ============================================================================

struct WorldStreamer::Impl {
    js5::Js5Cache cache;
    std::unordered_map<int32_t, UnderlayType> underlayTypes;
    UnderlayType defaultUnderlay;
    std::unordered_map<int32_t, OverlayType> overlayTypes;
    std::unordered_map<int32_t, LocType> locTypes;
    std::unordered_map<int32_t, TextureInfo> textureInfo;
    std::vector<std::vector<uint8_t>> textureLayers;
    std::vector<int32_t> palette;
    // Lit models are cached across all streamed squares: the same tree or
    // fence appears thousands of times across the world.
    std::unordered_map<int64_t, LitModel> litCache;
    size_t skippedModels = 0;

    void open(const std::string& cachePath);
    int32_t terrainGroupId(int mapX, int mapY);
    bool buildSquare(int mapX, int mapY, Model& out);

    const UnderlayType& underlay(int32_t id) const {
        auto it = underlayTypes.find(id);
        return it == underlayTypes.end() ? defaultUnderlay : it->second;
    }
};

void WorldStreamer::Impl::open(const std::string& cachePath) {
    cache.open(cachePath);

    for (const auto& [fileId, data] : cache.readGroupFiles(kConfigsIndex, kUnderlaysGroup)) {
        js5::ByteReader r(data);
        UnderlayType type;
        type.decode(r);
        underlayTypes.emplace(fileId, type);
    }
    defaultUnderlay.hsl = rgbToHslParts(0);

    for (const auto& [fileId, data] : cache.readGroupFiles(kConfigsIndex, kOverlaysGroup)) {
        js5::ByteReader r(data);
        OverlayType type;
        type.decode(r);
        overlayTypes.emplace(fileId, type);
    }

    for (const auto& [fileId, data] : cache.readGroupFiles(kConfigsIndex, 6)) {
        locTypes.emplace(fileId, decodeLocType(data));
    }

    // Texture definitions (simplified format, rev 233+: u16 sprite id,
    // u16 average HSL, ...) name sprites that become texture-array layers;
    // the average HSL stays as the fallback color.
    for (const auto& [fileId, data] : cache.readGroupFiles(kTexturesIndex, 0)) {
        if (data.size() < 4) continue;
        js5::ByteReader r(data);
        const int32_t spriteId = static_cast<int32_t>(r.u16());
        TextureInfo info;
        info.avgHsl = static_cast<int32_t>(r.u16());
        try {
            auto spriteFiles = cache.readGroupFiles(kSpritesIndex, spriteId);
            if (!spriteFiles.empty()) {
                std::vector<uint8_t> rgba;
                if (decodeSpriteRgba(spriteFiles.begin()->second, 0.8, rgba)) {
                    info.layer = static_cast<int32_t>(textureLayers.size());
                    textureLayers.push_back(std::move(rgba));
                }
            }
        } catch (const std::exception&) {
            // Missing sprite: the average-color fallback covers it.
        }
        textureInfo.emplace(fileId, info);
    }

    palette = buildPalette(0.8);

    std::printf("World data: %zu underlays, %zu overlays, %zu locs, %zu textures\n",
                underlayTypes.size(), overlayTypes.size(), locTypes.size(),
                textureLayers.size());
}

int32_t WorldStreamer::Impl::terrainGroupId(int mapX, int mapY) {
    if (mapX < 0 || mapY < 0 || mapX > 255 || mapY > 255) return -1;
    // Two map-index generations: older caches name their groups
    // ("m<x>_<y>", looked up by hash), newer ones key groups directly by
    // region id with terrain as file 0 and locs as file 1.
    if (cache.table(kMapsIndex).named) {
        return cache.groupIdForName(kMapsIndex, "m" + std::to_string(mapX) + "_" +
                                                    std::to_string(mapY));
    }
    const int32_t regionId = (mapX << 8) + mapY;
    return cache.table(kMapsIndex).group(regionId) != nullptr ? regionId : -1;
}

bool WorldStreamer::Impl::buildSquare(int mapX, int mapY, Model& out) {
    if (terrainGroupId(mapX, mapY) == -1) return false;

    // Decode the square plus its 8 neighbours: the blend kernel (radius 5)
    // and the lighting (radius 1) then see the same neighbourhood that the
    // adjacent squares' builds see, which keeps chunk borders seamless.
    constexpr int kSpan = 3;
    const int size = kSpan * kMapSquareSize;
    const int originTileX = (mapX - 1) * kMapSquareSize;
    const int originTileY = (mapY - 1) * kMapSquareSize;
    Scene scene(size, size);
    std::vector<LocPlacement> placements;

    for (int dx = 0; dx < kSpan; ++dx) {
        for (int dy = 0; dy < kSpan; ++dy) {
            const int mx = mapX - 1 + dx;
            const int my = mapY - 1 + dy;
            const int offsetX = dx * kMapSquareSize;
            const int offsetY = dy * kMapSquareSize;
            const int32_t groupId = terrainGroupId(mx, my);
            if (groupId == -1) {
                // No map data (ocean, unreleased): fill heights with the same
                // noise the client uses so lighting at the edge is correct;
                // the tiles stay untyped and are never rendered.
                for (int x = 0; x < kMapSquareSize; ++x) {
                    for (int y = 0; y < kMapSquareSize; ++y) {
                        scene.at(scene.heights, 0, offsetX + x, offsetY + y) =
                            -generateHeight(originTileX + offsetX + x + 932731,
                                            originTileY + offsetY + y + 556238) *
                            kHeightBasis;
                        for (int level = 1; level < kMaxLevels; ++level) {
                            scene.at(scene.heights, level, offsetX + x, offsetY + y) =
                                scene.get(scene.heights, level - 1, offsetX + x,
                                          offsetY + y) -
                                kLevelHeight;
                        }
                    }
                }
                continue;
            }
            auto files = cache.readGroupFiles(kMapsIndex, groupId);
            auto it = files.find(0);
            if (it == files.end()) continue;
            decodeTerrainSquare(scene, it->second, offsetX, offsetY, originTileX,
                                originTileY);
            if (dx != 1 || dy != 1) continue;

            // Loc placements: only the center square's belong to this chunk.
            if (cache.table(kMapsIndex).named) {
                const int32_t locGroup = cache.groupIdForName(
                    kMapsIndex, "l" + std::to_string(mx) + "_" + std::to_string(my));
                if (locGroup != -1) {
                    auto locFiles = cache.readGroupFiles(kMapsIndex, locGroup);
                    if (auto locIt = locFiles.find(0); locIt != locFiles.end()) {
                        decodeLocPlacements(locIt->second, offsetX, offsetY, placements);
                    }
                }
            } else if (auto locIt = files.find(1); locIt != files.end()) {
                decodeLocPlacements(locIt->second, offsetX, offsetY, placements);
            }
        }
    }

    // --- tile geometry for the center square --------------------------------
    Model& model = out;
    const glm::vec3 origin(static_cast<float>(originTileX), 0.0f,
                           static_cast<float>(-originTileY));

    for (int level = 0; level < kMaxLevels; ++level) {
        const std::vector<int32_t> blended =
            blendUnderlays(scene, level, underlayTypes, defaultUnderlay);
        const std::vector<int32_t> lights = calculateTileLights(scene, level);

        auto lightAt = [&](int x, int y) {
            return lights[static_cast<size_t>(x) * scene.sizeY + y];
        };
        auto blendedAt = [&](int x, int y) {
            return blended[static_cast<size_t>(x) * scene.sizeY + y];
        };

        for (int x = kMapSquareSize; x < 2 * kMapSquareSize; ++x) {
            for (int y = kMapSquareSize; y < 2 * kMapSquareSize; ++y) {
                const int32_t underlayId = scene.get(scene.underlays, level, x, y) - 1;
                const int32_t overlayId =
                    (scene.get(scene.overlays, level, x, y) & 0x7fff) - 1;
                if (underlayId == -1 && overlayId == -1) continue;

                const int32_t heightSw = scene.get(scene.heights, level, x, y);
                const int32_t heightSe = scene.get(scene.heights, level, x + 1, y);
                const int32_t heightNe = scene.get(scene.heights, level, x + 1, y + 1);
                const int32_t heightNw = scene.get(scene.heights, level, x, y + 1);

                const int32_t lightSw = lightAt(x, y);
                const int32_t lightSe = lightAt(x + 1, y);
                const int32_t lightNe = lightAt(x + 1, y + 1);
                const int32_t lightNw = lightAt(x, y + 1);

                // Blended underlay color; the client only smooths across
                // corners in HD mode, so all four corners share the SW color.
                int32_t blendHsl = -1;
                if (underlayId != -1) blendHsl = blendedAt(x, y);

                int shape = 0;
                int rotation = 0;
                int32_t overlayHsl = 0;
                int32_t overlayLayer = -1;
                if (overlayId != -1) {
                    shape = scene.get(scene.shapes, level, x, y) + 1;
                    rotation = scene.get(scene.rotations, level, x, y);

                    static const OverlayType defaultOverlay{};
                    auto overlayIt = overlayTypes.find(overlayId);
                    const OverlayType& overlay =
                        overlayIt == overlayTypes.end() ? defaultOverlay : overlayIt->second;

                    auto textureIt = textureInfo.find(overlay.textureId);
                    if (overlay.textureId != -1 && textureIt != textureInfo.end()) {
                        if (textureIt->second.layer >= 0) {
                            // Sampled texture: vertices carry light only
                            // (overlayHsl -1 means exactly that downstream).
                            overlayLayer = textureIt->second.layer;
                            overlayHsl = -1;
                        } else {
                            overlayHsl = textureIt->second.avgHsl;
                        }
                    } else if (overlay.primaryRgb == 0xff00ff) {
                        overlayHsl = -2; // invisible marker overlay
                    } else {
                        overlayHsl = packHsl(overlay.hsl.hue256, overlay.hsl.saturation,
                                             overlay.hsl.lightness);
                    }
                }

                // Per-corner lit colors.
                const int32_t underSw = adjustUnderlayLight(blendHsl, lightSw);
                const int32_t underSe = adjustUnderlayLight(blendHsl, lightSe);
                const int32_t underNe = adjustUnderlayLight(blendHsl, lightNe);
                const int32_t underNw = adjustUnderlayLight(blendHsl, lightNw);
                const int32_t overSw = adjustOverlayLight(overlayHsl, lightSw);
                const int32_t overSe = adjustOverlayLight(overlayHsl, lightSe);
                const int32_t overNe = adjustOverlayLight(overlayHsl, lightNe);
                const int32_t overNw = adjustOverlayLight(overlayHsl, lightNw);

                // Template vertices for this tile's shape.
                const std::vector<int>& vertexIndices =
                    kShapeVertexIndices[static_cast<size_t>(shape)];
                const size_t vertexCount = vertexIndices.size();
                std::array<int32_t, 6> vertX{};
                std::array<int32_t, 6> vertY{};
                std::array<int32_t, 6> vertZ{};
                std::array<int32_t, 6> vertUnderHsl{};
                std::array<int32_t, 6> vertOverHsl{};

                const int32_t tileX = x * kTileUnits;
                const int32_t tileY = y * kTileUnits;
                constexpr int32_t kHalf = kTileUnits / 2;
                constexpr int32_t kQuarter = kTileUnits / 4;
                constexpr int32_t kThreeQtr = kTileUnits * 3 / 4;

                for (size_t i = 0; i < vertexCount; ++i) {
                    int vertexIndex = vertexIndices[i];
                    if ((vertexIndex & 1) == 0 && vertexIndex <= 8) {
                        vertexIndex = ((vertexIndex - rotation - rotation - 1) & 7) + 1;
                    }
                    if (vertexIndex > 8 && vertexIndex <= 12) {
                        vertexIndex = ((vertexIndex - 9 - rotation) & 3) + 9;
                    }
                    if (vertexIndex > 12 && vertexIndex <= 16) {
                        vertexIndex = ((vertexIndex - 13 - rotation) & 3) + 13;
                    }

                    int32_t vx;
                    int32_t vz;
                    int32_t vy;
                    int32_t underHsl;
                    int32_t overHsl;
                    switch (vertexIndex) {
                        case 1:
                            vx = tileX; vz = tileY; vy = heightSw;
                            underHsl = underSw; overHsl = overSw;
                            break;
                        case 2:
                            vx = tileX + kHalf; vz = tileY;
                            vy = (heightSe + heightSw) >> 1;
                            underHsl = mixHsl(underSe, underSw);
                            overHsl = (overSe + overSw) >> 1;
                            break;
                        case 3:
                            vx = tileX + kTileUnits; vz = tileY; vy = heightSe;
                            underHsl = underSe; overHsl = overSe;
                            break;
                        case 4:
                            vx = tileX + kTileUnits; vz = tileY + kHalf;
                            vy = (heightNe + heightSe) >> 1;
                            underHsl = mixHsl(underSe, underNe);
                            overHsl = (overSe + overNe) >> 1;
                            break;
                        case 5:
                            vx = tileX + kTileUnits; vz = tileY + kTileUnits; vy = heightNe;
                            underHsl = underNe; overHsl = overNe;
                            break;
                        case 6:
                            vx = tileX + kHalf; vz = tileY + kTileUnits;
                            vy = (heightNe + heightNw) >> 1;
                            underHsl = mixHsl(underNw, underNe);
                            overHsl = (overNw + overNe) >> 1;
                            break;
                        case 7:
                            vx = tileX; vz = tileY + kTileUnits; vy = heightNw;
                            underHsl = underNw; overHsl = overNw;
                            break;
                        case 8:
                            vx = tileX; vz = tileY + kHalf;
                            vy = (heightNw + heightSw) >> 1;
                            underHsl = mixHsl(underNw, underSw);
                            overHsl = (overNw + overSw) >> 1;
                            break;
                        case 9:
                            vx = tileX + kHalf; vz = tileY + kQuarter;
                            vy = (heightSe + heightSw) >> 1;
                            underHsl = mixHsl(underSe, underSw);
                            overHsl = (overSe + overSw) >> 1;
                            break;
                        case 10:
                            vx = tileX + kThreeQtr; vz = tileY + kHalf;
                            vy = (heightNe + heightSe) >> 1;
                            underHsl = mixHsl(underSe, underNe);
                            overHsl = (overSe + overNe) >> 1;
                            break;
                        case 11:
                            vx = tileX + kHalf; vz = tileY + kThreeQtr;
                            vy = (heightNe + heightNw) >> 1;
                            underHsl = mixHsl(underNw, underNe);
                            overHsl = (overNw + overNe) >> 1;
                            break;
                        case 12:
                            vx = tileX + kQuarter; vz = tileY + kHalf;
                            vy = (heightNw + heightSw) >> 1;
                            underHsl = mixHsl(underNw, underSw);
                            overHsl = (overNw + overSw) >> 1;
                            break;
                        case 13:
                            vx = tileX + kQuarter; vz = tileY + kQuarter; vy = heightSw;
                            underHsl = underSw; overHsl = overSw;
                            break;
                        case 14:
                            vx = tileX + kThreeQtr; vz = tileY + kQuarter; vy = heightSe;
                            underHsl = underSe; overHsl = overSe;
                            break;
                        case 15:
                            vx = tileX + kThreeQtr; vz = tileY + kThreeQtr; vy = heightNe;
                            underHsl = underNe; overHsl = overNe;
                            break;
                        default:
                            vx = tileX + kQuarter; vz = tileY + kThreeQtr; vy = heightNw;
                            underHsl = underNw; overHsl = overNw;
                            break;
                    }

                    vertX[i] = vx;
                    vertY[i] = vy;
                    vertZ[i] = vz;
                    vertUnderHsl[i] = underHsl;
                    vertOverHsl[i] = overHsl;
                }

                // Faces.
                const std::vector<int>& faces = kShapeFaces[static_cast<size_t>(shape)];
                for (size_t f = 0; f + 3 < faces.size(); f += 4) {
                    const bool isOverlay = faces[f] == 1;
                    int a = faces[f + 1];
                    int b = faces[f + 2];
                    int c = faces[f + 3];
                    if (a < 4) a = (a - rotation) & 3;
                    if (b < 4) b = (b - rotation) & 3;
                    if (c < 4) c = (c - rotation) & 3;

                    const int32_t hslA = isOverlay ? vertOverHsl[static_cast<size_t>(a)]
                                                   : vertUnderHsl[static_cast<size_t>(a)];
                    const int32_t hslB = isOverlay ? vertOverHsl[static_cast<size_t>(b)]
                                                   : vertUnderHsl[static_cast<size_t>(b)];
                    const int32_t hslC = isOverlay ? vertOverHsl[static_cast<size_t>(c)]
                                                   : vertUnderHsl[static_cast<size_t>(c)];
                    if (hslA == kInvalidHsl) continue; // hidden face

                    auto toViewer = [&](size_t vi) {
                        return origin + glm::vec3(static_cast<float>(vertX[vi]) / kTileUnits,
                                                  -static_cast<float>(vertY[vi]) / kTileUnits,
                                                  -static_cast<float>(vertZ[vi]) / kTileUnits);
                    };
                    // Textured overlays (water, roads) span the texture once
                    // per tile: UVs are the vertex position within the tile.
                    const bool texturedFace = isOverlay && overlayLayer >= 0;
                    glm::vec2 tileUvs[3];
                    if (texturedFace) {
                        const int idx[3] = {a, b, c};
                        for (int k = 0; k < 3; ++k) {
                            tileUvs[k] = glm::vec2(
                                static_cast<float>(vertX[static_cast<size_t>(idx[k])] - tileX) /
                                    kTileUnits,
                                static_cast<float>(vertZ[static_cast<size_t>(idx[k])] - tileY) /
                                    kTileUnits);
                        }
                    }
                    emitTriangle(model, palette, toViewer(static_cast<size_t>(a)),
                                 toViewer(static_cast<size_t>(b)),
                                 toViewer(static_cast<size_t>(c)), hslA, hslB, hslC,
                                 /*reorientUp=*/true, texturedFace ? tileUvs : nullptr,
                                 texturedFace ? overlayLayer : -1);
                }
            }
        }
    }

    // --- loc scenery: stamp lit models onto the terrain ----------------------
    auto heightAt = [&](int level, int x, int y) {
        x = std::clamp(x, 0, scene.sizeX - 1);
        y = std::clamp(y, 0, scene.sizeY - 1);
        return scene.get(scene.heights, level, x, y);
    };

    for (const LocPlacement& placement : placements) {
        auto typeIt = locTypes.find(placement.id);
        if (typeIt == locTypes.end()) continue;
        const LocType& type = typeIt->second;

        // Pick the model list for this placement's shape; a shape of -1 in
        // the config means "whatever shape the map asks for".
        const std::vector<int32_t>* modelIds = nullptr;
        for (const auto& [shape, ids] : type.shapeModels) {
            if (shape == placement.shape) {
                modelIds = &ids;
                break;
            }
        }
        if (modelIds == nullptr) {
            for (const auto& [shape, ids] : type.shapeModels) {
                if (shape == -1) {
                    modelIds = &ids;
                    break;
                }
            }
        }
        if (modelIds == nullptr || modelIds->empty()) continue;

        // Decode + light once per (loc, shape), across all squares.
        const int64_t cacheKey = static_cast<int64_t>(placement.id) * 64 + placement.shape;
        auto litIt = litCache.find(cacheKey);
        if (litIt == litCache.end()) {
            RawModel merged;
            for (const int32_t modelId : *modelIds) {
                try {
                    auto files = cache.readGroupFiles(7, modelId);
                    if (files.empty()) continue;
                    RawModel part;
                    if (!decodeModel(files.begin()->second, part)) {
                        ++skippedModels; // unported format variant
                        continue;
                    }
                    const int32_t vertexBase = merged.vertexCount;
                    const int32_t texBase = merged.texTriangleCount;
                    merged.vertexCount += part.vertexCount;
                    merged.vx.insert(merged.vx.end(), part.vx.begin(), part.vx.end());
                    merged.vy.insert(merged.vy.end(), part.vy.begin(), part.vy.end());
                    merged.vz.insert(merged.vz.end(), part.vz.begin(), part.vz.end());
                    // Per-face optional streams must stay aligned across the
                    // merge: if either side has one, pad the faces merged so
                    // far with defaults, then append this part's values.
                    const bool needTypes =
                        !part.faceRenderTypes.empty() || !merged.faceRenderTypes.empty();
                    const bool needAlphas =
                        !part.faceAlphas.empty() || !merged.faceAlphas.empty();
                    const bool needTextures =
                        !part.faceTextures.empty() || !merged.faceTextures.empty();
                    const bool needCoords =
                        !part.textureCoords.empty() || !merged.textureCoords.empty();
                    if (needTypes) merged.faceRenderTypes.resize(static_cast<size_t>(merged.faceCount), 0);
                    if (needAlphas) merged.faceAlphas.resize(static_cast<size_t>(merged.faceCount), 0);
                    if (needTextures) merged.faceTextures.resize(static_cast<size_t>(merged.faceCount), -1);
                    if (needCoords) merged.textureCoords.resize(static_cast<size_t>(merged.faceCount), -1);
                    for (int32_t f = 0; f < part.faceCount; ++f) {
                        const size_t pf = static_cast<size_t>(f);
                        merged.ia.push_back(part.ia[pf] + vertexBase);
                        merged.ib.push_back(part.ib[pf] + vertexBase);
                        merged.ic.push_back(part.ic[pf] + vertexBase);
                        merged.faceColors.push_back(part.faceColors[pf]);
                        if (needTypes) {
                            merged.faceRenderTypes.push_back(
                                part.faceRenderTypes.empty() ? 0 : part.faceRenderTypes[pf]);
                        }
                        if (needAlphas) {
                            merged.faceAlphas.push_back(
                                part.faceAlphas.empty() ? 0 : part.faceAlphas[pf]);
                        }
                        if (needTextures) {
                            merged.faceTextures.push_back(
                                part.faceTextures.empty() ? int16_t{-1} : part.faceTextures[pf]);
                        }
                        if (needCoords) {
                            int8_t coord =
                                part.textureCoords.empty() ? int8_t{-1} : part.textureCoords[pf];
                            if (coord != -1) {
                                // Coord indices are per-part; rebase them.
                                const int32_t rebased = (coord & 0xff) + texBase;
                                coord = rebased <= 127 ? static_cast<int8_t>(rebased)
                                                       : int8_t{-1};
                            }
                            merged.textureCoords.push_back(coord);
                        }
                    }
                    merged.faceCount += part.faceCount;
                    // Texture-mapping triangles reference part-local vertices.
                    for (int32_t t = 0; t < part.texTriangleCount; ++t) {
                        const size_t pt = static_cast<size_t>(t);
                        const bool valid = part.texP[pt] != 0xffff;
                        merged.texP.push_back(valid ? static_cast<uint16_t>(part.texP[pt] + vertexBase) : uint16_t{0xffff});
                        merged.texM.push_back(valid ? static_cast<uint16_t>(part.texM[pt] + vertexBase) : uint16_t{0xffff});
                        merged.texN.push_back(valid ? static_cast<uint16_t>(part.texN[pt] + vertexBase) : uint16_t{0xffff});
                    }
                    merged.texTriangleCount += part.texTriangleCount;
                } catch (const std::exception&) {
                    ++skippedModels;
                }
            }

            for (const auto& [from, to] : type.recolor) {
                for (uint16_t& color : merged.faceColors) {
                    if (color == from) color = to;
                }
            }
            for (const auto& [from, to] : type.retexture) {
                for (int16_t& texture : merged.faceTextures) {
                    if (texture == static_cast<int16_t>(from)) {
                        texture = static_cast<int16_t>(to);
                    }
                }
            }

            LitModel lit;
            if (merged.faceCount > 0) {
                lightModel(merged, 64 + type.ambient, 768 + type.contrast, textureInfo,
                           lit);
            }
            litIt = litCache.emplace(cacheKey, std::move(lit)).first;
        }
        const LitModel& lit = litIt->second;
        if (lit.positions.empty()) continue;

        // Placement transform: uniform per-axis scale, N x 90-degree yaw,
        // config offset, then translation to the loc's center on its tile.
        const bool swapSize = (placement.rotation & 1) != 0;
        const int32_t tilesX = swapSize ? type.sizeY : type.sizeX;
        const int32_t tilesY = swapSize ? type.sizeX : type.sizeY;
        const int32_t baseUnitsX = placement.x * kTileUnits + tilesX * (kTileUnits / 2);
        const int32_t baseUnitsZ = placement.y * kTileUnits + tilesY * (kTileUnits / 2);
        const int32_t height =
            (heightAt(placement.level, placement.x, placement.y) +
             heightAt(placement.level, placement.x + tilesX, placement.y) +
             heightAt(placement.level, placement.x, placement.y + tilesY) +
             heightAt(placement.level, placement.x + tilesX, placement.y + tilesY)) / 4;

        auto transform = [&](glm::ivec3 v) {
            v.x = v.x * type.modelSizeX / 128;
            v.y = v.y * type.modelSizeHeight / 128;
            v.z = v.z * type.modelSizeY / 128;
            for (int r = 0; r < placement.rotation; ++r) {
                const int32_t tmp = v.x;
                v.x = v.z;
                v.z = -tmp;
            }
            v.x += type.offsetX;
            v.y += type.offsetHeight;
            v.z += type.offsetY;
            return origin + glm::vec3(static_cast<float>(baseUnitsX + v.x) / kTileUnits,
                                      -static_cast<float>(height + v.y) / kTileUnits,
                                      -static_cast<float>(baseUnitsZ + v.z) / kTileUnits);
        };

        for (size_t f = 0; f + 2 < lit.positions.size(); f += 3) {
            emitTriangle(model, palette, transform(lit.positions[f]),
                         transform(lit.positions[f + 1]), transform(lit.positions[f + 2]),
                         lit.hsls[f], lit.hsls[f + 1], lit.hsls[f + 2],
                         /*reorientUp=*/false, &lit.uvs[f], lit.layers[f / 3]);
        }
    }

    return true;
}

WorldStreamer::WorldStreamer() : impl_(std::make_unique<Impl>()) {}
WorldStreamer::~WorldStreamer() = default;

void WorldStreamer::open(const std::string& cachePath) {
    impl_->open(cachePath);
}

const std::vector<std::vector<uint8_t>>& WorldStreamer::textures() const {
    return impl_->textureLayers;
}

bool WorldStreamer::buildSquare(int mapX, int mapY, Model& out) {
    return impl_->buildSquare(mapX, mapY, out);
}
