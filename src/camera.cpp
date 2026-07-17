#include "camera.h"

// GLM_FORCE_DEPTH_ZERO_TO_ONE and GLM_FORCE_RADIANS are defined project-wide
// in CMakeLists.txt. The depth one is load-bearing: GLM defaults to OpenGL's
// [-1, 1] clip-space depth, but Vulkan (like D3D and Metal) uses [0, 1].
// Without the define, glm::perspective would build a matrix that wastes half
// the depth range and breaks near-plane clipping. It must be set globally so
// every translation unit sees the same glm::perspective.
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>

namespace {

constexpr float kRotateSpeed = 0.30f;  // degrees per pixel of drag
constexpr float kZoomPerStep = 0.88f;  // distance multiplier per wheel step
constexpr float kMinPitch = -85.0f;    // stop short of the poles so the
constexpr float kMaxPitch = 85.0f;     // lookAt up-vector never degenerates
constexpr float kMinDistance = 2.0f;
constexpr float kMaxDistance = 1500.0f; // far enough to frame a loaded world area
constexpr float kPanSpeed = 0.0016f;    // fraction of distance per pixel of drag
constexpr float kFovYDegrees = 50.0f;
constexpr float kNearPlane = 0.1f;
constexpr float kFarPlane = 4000.0f;

} // namespace

void OrbitCamera::rotate(float dx, float dy) {
    yawDeg_ += dx * kRotateSpeed;
    // Screen +y is down, so dragging up (negative dy) should raise the eye.
    pitchDeg_ = std::clamp(pitchDeg_ - dy * kRotateSpeed, kMinPitch, kMaxPitch);
}

void OrbitCamera::zoom(float steps) {
    distance_ = std::clamp(distance_ * std::pow(kZoomPerStep, steps),
                           kMinDistance, kMaxDistance);
}

void OrbitCamera::pan(float dx, float dy) {
    const float yaw = glm::radians(yawDeg_);
    // Camera right and (ground-projected) forward, derived from yaw only so
    // panning always slides along the ground plane regardless of pitch.
    const glm::vec3 right{std::cos(yaw), 0.0f, -std::sin(yaw)};
    const glm::vec3 forward{-std::sin(yaw), 0.0f, -std::cos(yaw)};
    const float scale = distance_ * kPanSpeed;
    target_ += right * (-dx * scale) + forward * (dy * scale);
}

void OrbitCamera::setAngles(float yawDeg, float pitchDeg) {
    yawDeg_ = yawDeg;
    pitchDeg_ = std::clamp(pitchDeg, kMinPitch, kMaxPitch);
}

void OrbitCamera::setDistance(float distance) {
    distance_ = std::clamp(distance, kMinDistance, kMaxDistance);
}

void OrbitCamera::setTarget(const glm::vec3& target) {
    target_ = target;
}

glm::mat4 OrbitCamera::viewMatrix() const {
    const float yaw = glm::radians(yawDeg_);
    const float pitch = glm::radians(pitchDeg_);
    const glm::vec3 offset{
        std::cos(pitch) * std::sin(yaw),
        std::sin(pitch),
        std::cos(pitch) * std::cos(yaw),
    };
    return glm::lookAt(target_ + offset * distance_, target_,
                       glm::vec3(0.0f, 1.0f, 0.0f));
}

glm::mat4 OrbitCamera::projectionMatrix(float aspect) const {
    glm::mat4 proj = glm::perspective(glm::radians(kFovYDegrees), aspect,
                                      kNearPlane, kFarPlane);

    // WebGL2 -> Vulkan translation, part 2 of the clip-space story: besides
    // the depth range handled by GLM_FORCE_DEPTH_ZERO_TO_ONE, Vulkan's clip
    // space has +Y pointing DOWN (framebuffer row 0 is the top row), where
    // GL had +Y up. Negating the Y scale here is the simplest fix. The
    // common alternative is a negative-height viewport (core since 1.1 via
    // VK_KHR_maintenance1), which flips at rasterization time instead;
    // we flip in the matrix so the shader math stays identical to the
    // WebGL2 renderer this is ported from. Consequence: triangle winding
    // appears mirrored on screen, which the pipeline's front-face setting
    // accounts for (see createGraphicsPipeline in renderer.cpp).
    proj[1][1] *= -1.0f;
    return proj;
}
