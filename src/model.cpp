#include "model.h"

#include <cmath>
#include <utility>

namespace {

constexpr float kTau = 6.28318530717958647692f;

// Tiny deterministic LCG (numerical recipes constants) used to jitter face
// colors. Deterministic on purpose: every run of the viewer produces the
// exact same model, which keeps screenshots and GPU captures reproducible.
class Lcg {
public:
    explicit Lcg(uint32_t seed) : state_(seed) {}

    // Uniform float in [0, 1).
    float next() {
        state_ = state_ * 1664525u + 1013904223u;
        return static_cast<float>(state_ >> 8) / 16777216.0f;
    }

    float range(float lo, float hi) { return lo + (hi - lo) * next(); }

private:
    uint32_t state_;
};

// Appends one flat-shaded triangle: three fresh vertices sharing the face
// normal and face color, plus three indices.
//
// `insidePoint` disambiguates winding: the face normal is forced to point
// away from it, so the mesh ends up consistently CCW-from-outside no matter
// how the caller ordered the corners. That matters because the pipeline
// culls back faces - a flipped triangle doesn't render wrong, it vanishes.
void addFace(Model& model, glm::vec3 a, glm::vec3 b, glm::vec3 c,
             glm::vec3 insidePoint, glm::vec3 color) {
    glm::vec3 normal = glm::cross(b - a, c - a);
    const float lengthSq = glm::dot(normal, normal);
    if (lengthSq < 1e-12f) return; // degenerate triangle, skip
    normal /= std::sqrt(lengthSq);

    if (glm::dot(normal, (a + b + c) / 3.0f - insidePoint) < 0.0f) {
        std::swap(b, c);
        normal = -normal;
    }

    color = glm::clamp(color, glm::vec3(0.0f), glm::vec3(1.0f));

    const auto base = static_cast<uint32_t>(model.vertices.size());
    model.vertices.push_back({a, normal, color});
    model.vertices.push_back({b, normal, color});
    model.vertices.push_back({c, normal, color});
    model.indices.push_back(base);
    model.indices.push_back(base + 1);
    model.indices.push_back(base + 2);
}

// A canopy "blob": an icosahedron (12 vertices, 20 faces) stretched to an
// ellipsoid. The classic low-poly sphere substitute - round enough to read
// as foliage, cheap enough that per-vertex lighting visibly facets it,
// which is exactly the OSRS aesthetic.
void addCanopyBlob(Model& model, glm::vec3 center, glm::vec3 radii,
                   glm::vec3 baseColor, Lcg& rng) {
    const float t = (1.0f + std::sqrt(5.0f)) / 2.0f; // golden ratio
    const glm::vec3 corners[12] = {
        {-1.0f, t, 0.0f},  {1.0f, t, 0.0f},  {-1.0f, -t, 0.0f}, {1.0f, -t, 0.0f},
        {0.0f, -1.0f, t},  {0.0f, 1.0f, t},  {0.0f, -1.0f, -t}, {0.0f, 1.0f, -t},
        {t, 0.0f, -1.0f},  {t, 0.0f, 1.0f},  {-t, 0.0f, -1.0f}, {-t, 0.0f, 1.0f},
    };
    static constexpr uint32_t faces[20][3] = {
        {0, 11, 5}, {0, 5, 1},  {0, 1, 7},   {0, 7, 10}, {0, 10, 11},
        {1, 5, 9},  {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
        {3, 9, 4},  {3, 4, 2},  {3, 2, 6},   {3, 6, 8},  {3, 8, 9},
        {4, 9, 5},  {2, 4, 11}, {6, 2, 10},  {8, 6, 7},  {9, 8, 1},
    };

    glm::vec3 positions[12];
    for (int i = 0; i < 12; ++i) {
        positions[i] = center + glm::normalize(corners[i]) * radii;
    }
    for (const auto& face : faces) {
        // Per-face tone jitter stands in for OSRS's dithered palette bands.
        const float tone = rng.range(0.82f, 1.18f);
        addFace(model, positions[face[0]], positions[face[1]], positions[face[2]],
                center, baseColor * tone);
    }
}

// Tapered hexagonal prism trunk. Open at both ends: the bottom sits on the
// ground disc and the top is buried inside the canopy, so caps would only
// add faces that can never be seen.
void addTrunk(Model& model, Lcg& rng) {
    constexpr int kSides = 6;
    constexpr float kBottomY = 0.0f;
    constexpr float kTopY = 1.8f;
    constexpr float kBottomRadius = 0.30f;
    constexpr float kTopRadius = 0.19f;
    const glm::vec3 brown{0.37f, 0.26f, 0.15f};
    const glm::vec3 axisMid{0.0f, (kBottomY + kTopY) * 0.5f, 0.0f};

    for (int i = 0; i < kSides; ++i) {
        const float a0 = kTau * static_cast<float>(i) / kSides;
        const float a1 = kTau * static_cast<float>(i + 1) / kSides;
        const glm::vec3 b0{std::cos(a0) * kBottomRadius, kBottomY, std::sin(a0) * kBottomRadius};
        const glm::vec3 b1{std::cos(a1) * kBottomRadius, kBottomY, std::sin(a1) * kBottomRadius};
        const glm::vec3 t0{std::cos(a0) * kTopRadius, kTopY, std::sin(a0) * kTopRadius};
        const glm::vec3 t1{std::cos(a1) * kTopRadius, kTopY, std::sin(a1) * kTopRadius};

        // Each side is a quad split into two triangles, each with its own
        // bark tone so the trunk doesn't read as a single flat column.
        addFace(model, b0, b1, t1, axisMid, brown * rng.range(0.85f, 1.15f));
        addFace(model, b0, t1, t0, axisMid, brown * rng.range(0.85f, 1.15f));
    }
}

// A small grass disc under the trunk. Top-facing only; the underside is
// back-face culled, which is fine - the camera never goes underground.
void addGroundDisc(Model& model, Lcg& rng) {
    constexpr int kSegments = 8;
    constexpr float kRadius = 1.35f;
    const glm::vec3 grass{0.20f, 0.31f, 0.12f};
    const glm::vec3 center{0.0f, 0.0f, 0.0f};
    const glm::vec3 below{0.0f, -1.0f, 0.0f}; // forces normals to face up

    for (int i = 0; i < kSegments; ++i) {
        const float a0 = kTau * static_cast<float>(i) / kSegments;
        const float a1 = kTau * static_cast<float>(i + 1) / kSegments;
        const glm::vec3 r0{std::cos(a0) * kRadius, 0.0f, std::sin(a0) * kRadius};
        const glm::vec3 r1{std::cos(a1) * kRadius, 0.0f, std::sin(a1) * kRadius};
        addFace(model, center, r0, r1, below, grass * rng.range(0.88f, 1.12f));
    }
}

} // namespace

Model loadModel() {
    // Placeholder: a low-poly tree in the spirit of OSRS's classic "Tree"
    // (trunk + lumpy three-blob canopy), ~200 faces. See model.h for the
    // contract a real OSRS cache decoder has to fulfil to replace this.
    Model model;
    Lcg rng(0x05EED1E5u);

    addGroundDisc(model, rng);
    addTrunk(model, rng);
    addCanopyBlob(model, {0.00f, 2.45f, 0.00f}, {1.05f, 0.85f, 1.05f},
                  {0.20f, 0.42f, 0.12f}, rng);
    addCanopyBlob(model, {0.62f, 2.02f, 0.30f}, {0.62f, 0.55f, 0.62f},
                  {0.16f, 0.36f, 0.10f}, rng);
    addCanopyBlob(model, {-0.55f, 2.10f, -0.30f}, {0.68f, 0.58f, 0.68f},
                  {0.24f, 0.45f, 0.14f}, rng);

    return model;
}
