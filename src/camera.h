#pragma once

#include <glm/glm.hpp>

// Orbit camera: the eye rides a sphere around a fixed target point.
// Mouse drag changes yaw/pitch, scroll changes the sphere radius.
//
// The camera is plain math with no Vulkan (or GLFW) dependency - it turns
// user input into a view and projection matrix, and main.cpp hands those to
// the renderer once per frame. This is the same split as the XRSPS WebGL2
// renderer, where the camera produced matrices for a uniform block.
class OrbitCamera {
public:
    // dx/dy are cursor deltas in pixels (screen space: +y is down).
    void rotate(float dx, float dy);

    // steps is the scroll wheel delta; positive zooms in. Multiplicative so
    // zoom speed feels constant at any distance.
    void zoom(float steps);

    // Translates the orbit target on the ground plane, in screen-relative
    // directions (drag right moves the world right). Scaled by distance so
    // panning covers more ground when zoomed out.
    void pan(float dx, float dy);

    void setAngles(float yawDeg, float pitchDeg);
    void setDistance(float distance);
    void setTarget(const glm::vec3& target);
    const glm::vec3& target() const { return target_; }

    glm::mat4 viewMatrix() const;

    // aspect = swapchain width / height. The matrix includes the Y flip that
    // maps GLM's OpenGL-convention clip space onto Vulkan's; see the .cpp.
    glm::mat4 projectionMatrix(float aspect) const;

private:
    float yawDeg_ = 38.0f;
    float pitchDeg_ = 22.0f;   // elevation above the horizon; positive = above
    float distance_ = 8.0f;
    glm::vec3 target_{0.0f, 1.35f, 0.0f}; // mid-height of the placeholder tree
};
