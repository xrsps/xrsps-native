#pragma once

#include <vulkan/vulkan.h>

#include <glm/glm.hpp>

#include <cstdint>
#include <string>
#include <vector>

#include "model.h"

struct GLFWwindow;

// Per-instance data, fed to the vertex shader through a second vertex buffer
// binding that advances once per *instance* instead of once per vertex
// (VK_VERTEX_INPUT_RATE_INSTANCE).
//
// WebGL2 note: identical concept to gl.vertexAttribDivisor(loc, 1) - the
// difference is that Vulkan bakes the divisor into the pipeline's vertex
// input state instead of latching it as global attribute state.
struct InstanceData {
    glm::mat4 model;
};

// Everything the renderer needs from the outside world to draw one frame.
// The camera lives in main.cpp; the renderer only ever sees matrices.
struct FrameInput {
    glm::mat4 view;
    glm::mat4 proj;
    uint32_t instanceCount = 1;
    uint32_t firstInstance = 0;
    // Lighting terms for the Gouraud vertex shader. World-terrain mode sets
    // (1, 0): the cache pipeline bakes its hillshade lighting into the vertex
    // colors, so the shader must pass them through unmodified.
    float ambient = 0.42f;
    float diffuse = 0.62f;
};

// Renderer owns every Vulkan object in the application and tears them down
// in reverse creation order.
//
// WebGL2 note: the closest WebGL2 equivalent to this whole class is the one
// line `canvas.getContext("webgl2")`. Everything the browser and driver did
// behind that call - picking a GPU, creating a device context, allocating a
// framebuffer chain for the canvas, deciding when to present - is an explicit
// object with an explicit lifetime below.
class Renderer {
public:
    // `textures` is an optional array of 128x128 RGBA images (sRGB bytes)
    // sampled by faces whose vertices carry a uvLayer; pass an empty vector
    // for untextured scenes (a 1-layer placeholder keeps descriptors valid).
    Renderer(GLFWwindow* window, const Model& model,
             const std::vector<InstanceData>& instances,
             const std::vector<std::vector<uint8_t>>& textures = {});
    ~Renderer();

    // Vulkan handles inside this class must never be duplicated or freed
    // twice, so the renderer is deliberately non-copyable.
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    // Renders and presents one frame. Returns false if the frame was skipped
    // (window minimized to 0x0, or the swapchain had to be recreated).
    bool drawFrame(const FrameInput& input);

    // --- streamed chunk geometry -------------------------------------------
    // World mode streams map squares in and out at runtime: each chunk is an
    // independent vertex/index buffer drawn alongside the base model with
    // the same pipeline (instance 0's identity transform). Returns 0 for an
    // empty model, otherwise a handle for removeChunkGeometry.
    uint64_t addChunkGeometry(const Model& model);
    // Removal is deferred until the frames that may still reference the
    // buffers have finished (see the retired list in drawFrame) - the GPU
    // owns buffers long after the CPU stops mentioning them.
    void removeChunkGeometry(uint64_t handle);

    // Called from the GLFW framebuffer-resize callback. Only sets a flag:
    // the swapchain is recreated at a well-defined point inside drawFrame,
    // never mid-frame from a callback.
    void onFramebufferResized() { framebufferResized_ = true; }

    // Saves the next presented frame to `path` as a binary PPM (P6).
    void requestScreenshot(std::string path);

    // Blocks until the GPU has finished all submitted work. Called before
    // shutdown so no object is destroyed while a queue still reads from it.
    void waitIdle();

    // Current swapchain size; main.cpp derives the projection aspect ratio
    // from this rather than from the window size (they can differ during a
    // resize, and the swapchain is what we actually render into).
    VkExtent2D swapchainExtent() const { return swapchainExtent_; }

    // Name of the physical device we ended up on (for logs / window title).
    const char* deviceName() const { return deviceProperties_.deviceName; }

    // Number of warnings + errors the validation layers have reported.
    // The app exits non-zero if this is not 0 - "runs validation-clean" is
    // treated as a hard correctness requirement, not a nice-to-have.
    uint64_t validationMessageCount() const { return validationMessages_; }

private:
    // --- construction stages, in call order --------------------------------
    void createInstance();
    void setupDebugMessenger();
    void createSurface();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createSwapchain();
    void createSwapchainImageViews();
    void createRenderPass();
    void createDepthResources();
    void createFramebuffers();
    void createDescriptorSetLayout();
    void createGraphicsPipeline();
    void createCommandPool();
    void createTextureArray(const std::vector<std::vector<uint8_t>>& textures);
    void createGeometryBuffers(const Model& model,
                               const std::vector<InstanceData>& instances);
    void createUniformBuffers();
    void createDescriptorSets();
    void allocateCommandBuffers();
    void createSyncObjects();
    void createRenderFinishedSemaphores();

    // --- swapchain lifecycle ------------------------------------------------
    void cleanupSwapchain();
    void recreateSwapchain();

    // --- per-frame work ----------------------------------------------------
    void updateUniformBuffer(uint32_t frameIndex, const FrameInput& input);
    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex,
                             const FrameInput& input, bool captureAfterRender);

    // --- screenshot --------------------------------------------------------
    void prepareScreenshotBuffer();
    void saveScreenshot();

    // --- helpers ------------------------------------------------------------
    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags properties) const;
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags properties, VkBuffer& buffer,
                      VkDeviceMemory& memory);
    void createDeviceLocalBuffer(const void* data, VkDeviceSize size,
                                 VkBufferUsageFlags usage, VkBuffer& buffer,
                                 VkDeviceMemory& memory);
    VkCommandBuffer beginOneShotCommands();
    void endOneShotCommands(VkCommandBuffer cmd);
    VkShaderModule loadShaderModule(const char* fileName);

    // Two frames may be "in flight" (recorded on the CPU / executing on the
    // GPU) at once; see createSyncObjects() for the full discussion.
    static constexpr uint32_t kMaxFramesInFlight = 2;

    GLFWwindow* window_ = nullptr;

    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;

    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties deviceProperties_{};
    uint32_t queueFamilyIndex_ = 0;
    VkFormat depthFormat_ = VK_FORMAT_UNDEFINED;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;

    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat swapchainFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D swapchainExtent_{};
    std::vector<VkImage> swapchainImages_;
    std::vector<VkImageView> swapchainImageViews_;
    bool supportsScreenshots_ = false;

    VkImage depthImage_ = VK_NULL_HANDLE;
    VkDeviceMemory depthMemory_ = VK_NULL_HANDLE;
    VkImageView depthView_ = VK_NULL_HANDLE;

    VkRenderPass renderPass_ = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> framebuffers_;

    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;

    VkCommandPool commandPool_ = VK_NULL_HANDLE;

    // Texture array sampled by world-mode geometry (or a 1-layer stand-in).
    VkImage textureImage_ = VK_NULL_HANDLE;
    VkDeviceMemory textureMemory_ = VK_NULL_HANDLE;
    VkImageView textureView_ = VK_NULL_HANDLE;
    VkSampler textureSampler_ = VK_NULL_HANDLE;

    VkBuffer vertexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory vertexMemory_ = VK_NULL_HANDLE;
    VkBuffer indexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory indexMemory_ = VK_NULL_HANDLE;
    uint32_t indexCount_ = 0;
    VkBuffer instanceBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory instanceMemory_ = VK_NULL_HANDLE;
    uint32_t instanceCapacity_ = 0;

    VkBuffer uniformBuffers_[kMaxFramesInFlight] = {};
    VkDeviceMemory uniformMemories_[kMaxFramesInFlight] = {};
    void* uniformMapped_[kMaxFramesInFlight] = {};

    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSets_[kMaxFramesInFlight] = {};

    VkCommandBuffer commandBuffers_[kMaxFramesInFlight] = {};

    // Per frame-in-flight: "the presentation engine handed us an image"
    // and "this frame's GPU work is done" (fence, so the CPU can wait).
    VkSemaphore imageAvailable_[kMaxFramesInFlight] = {};
    VkFence inFlight_[kMaxFramesInFlight] = {};
    // Per *swapchain image*: "rendering into this image finished, it may be
    // presented". See createRenderFinishedSemaphores() for why this set is
    // sized differently from the two above.
    std::vector<VkSemaphore> renderFinished_;

    uint32_t currentFrame_ = 0;
    uint64_t absoluteFrame_ = 0;
    bool framebufferResized_ = false;

    struct Chunk {
        uint64_t id = 0;
        VkBuffer vertexBuffer = VK_NULL_HANDLE;
        VkDeviceMemory vertexMemory = VK_NULL_HANDLE;
        VkBuffer indexBuffer = VK_NULL_HANDLE;
        VkDeviceMemory indexMemory = VK_NULL_HANDLE;
        uint32_t indexCount = 0;
        uint64_t retiredAtFrame = 0;
    };
    std::vector<Chunk> chunks_;
    std::vector<Chunk> retiredChunks_;
    uint64_t nextChunkId_ = 1;
    void destroyChunk(Chunk& chunk);

    VkBuffer screenshotBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory screenshotMemory_ = VK_NULL_HANDLE;
    std::string pendingScreenshotPath_;

    uint64_t validationMessages_ = 0;
};
