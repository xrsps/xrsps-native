#include "renderer.h"

// renderer.h pulls in <vulkan/vulkan.h> before this include, which is what
// makes glfw3.h expose its Vulkan entry points (glfwCreateWindowSurface and
// glfwGetRequiredInstanceExtensions).
#include <GLFW/glfw3.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Validation layers.
//
// WebGL2 is a validated API by specification: the browser checks every call
// and throws or logs on misuse, and you pay that cost on every call, always.
// Vulkan inverts the deal - the driver checks nothing at runtime, and the
// same checks are provided as a *layer* you inject between the app and the
// driver during development, then drop entirely for release builds.
// ---------------------------------------------------------------------------
#ifdef NDEBUG
constexpr bool kEnableValidation = false;
#else
constexpr bool kEnableValidation = true;
#endif

constexpr const char* kValidationLayerName = "VK_LAYER_KHRONOS_validation";

constexpr const char* kDeviceExtensions[] = {
    // Presentation is not core Vulkan; a device that can put pixels on a
    // screen advertises it as an extension. (Compute-only accelerators
    // legitimately don't.)
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

// Scene light direction: one directional light plus flat ambient, matching
// the terms the OSRS client bakes into vertex lighting. Direction is the way
// the light *travels* (roughly: sun high up, west-ish and behind the default
// camera). Ambient/diffuse strengths arrive per frame via FrameInput.
constexpr glm::vec3 kLightDirection{-0.45f, -1.0f, -0.55f};

// Dark blue-grey backdrop; values are linear because the sRGB attachment
// encodes on write.
constexpr VkClearColorValue kClearColor{{0.016f, 0.020f, 0.032f, 1.0f}};

// ---------------------------------------------------------------------------
// The per-frame uniform block, mirrored exactly by shaders/model.vert.
//
// std140 layout note: this struct is deliberately built from mat4s and
// vec4s only, which have identical size/alignment rules in C++ and std140,
// so the memcpy in updateUniformBuffer() is layout-safe. The classic
// footgun this dodges: a lone float after a vec3 lands at different offsets
// in C++ and std140. WebGL2 had the same std140 rules for uniform blocks -
// what's new in Vulkan is that nothing ever re-validates the match at
// runtime; get it wrong and you silently read garbage.
// ---------------------------------------------------------------------------
struct FrameUniforms {
    glm::mat4 view;
    glm::mat4 proj;
    glm::vec4 lightDirection;
    glm::vec4 lightParams; // x: ambient, y: diffuse, zw: unused padding
};
static_assert(sizeof(FrameUniforms) == 160, "must match the shader block");

void vkCheck(VkResult result, const char* what) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(what) + " failed (VkResult " +
                                 std::to_string(static_cast<int>(result)) + ")");
    }
}

bool validationLayerAvailable() {
    uint32_t count = 0;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> layers(count);
    vkEnumerateInstanceLayerProperties(&count, layers.data());
    for (const VkLayerProperties& layer : layers) {
        if (std::strcmp(layer.layerName, kValidationLayerName) == 0) return true;
    }
    return false;
}

// Every warning or error from the layers lands here. The message is printed
// and *counted*: main() turns a non-zero count into a non-zero exit code, so
// a validation regression fails loudly instead of scrolling past.
VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT /*severity*/,
    VkDebugUtilsMessageTypeFlagsEXT /*type*/,
    const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
    void* userData) {
    ++*static_cast<uint64_t*>(userData);
    std::fprintf(stderr, "[vulkan] %s\n", callbackData->pMessage);
    // Returning VK_FALSE tells the layer "do not abort the offending call";
    // we want to see the message but keep running.
    return VK_FALSE;
}

VkDebugUtilsMessengerCreateInfoEXT makeDebugMessengerCreateInfo(uint64_t* counter) {
    VkDebugUtilsMessengerCreateInfoEXT info{};
    info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    // Only warnings and errors: INFO/VERBOSE are useful when tracing a
    // specific problem but would make "zero messages" meaningless as a gate.
    info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    info.pfnUserCallback = debugCallback;
    info.pUserData = counter;
    return info;
}

// Returns the index of a queue family that supports BOTH graphics and
// presenting to `surface`, or UINT32_MAX if none exists.
//
// Design decision: we require a single combined family instead of handling
// split graphics/present families. Every desktop driver (NVIDIA, AMD, Intel,
// Mesa) exposes a combined family, so the split path would be untestable
// dead code here. The alternative - two families, resources created with
// VK_SHARING_MODE_CONCURRENT or ownership transfers, and a second queue for
// vkQueuePresentKHR - is named in STUDY.md for completeness.
uint32_t findGraphicsPresentFamily(VkPhysicalDevice device, VkSurfaceKHR surface) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());
    for (uint32_t i = 0; i < count; ++i) {
        if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) continue;
        VkBool32 canPresent = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &canPresent);
        if (canPresent == VK_TRUE) return i;
    }
    return UINT32_MAX;
}

bool hasDeviceExtensions(VkPhysicalDevice device) {
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> available(count);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count, available.data());
    for (const char* needed : kDeviceExtensions) {
        bool found = false;
        for (const VkExtensionProperties& ext : available) {
            if (std::strcmp(ext.extensionName, needed) == 0) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    return true;
}

// Directory containing the running executable; used to locate the compiled
// shaders regardless of the working directory the app was launched from.
std::filesystem::path executableDir() {
#ifdef _WIN32
    char buffer[MAX_PATH] = {};
    const DWORD n = GetModuleFileNameA(nullptr, buffer, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) return {};
    return std::filesystem::path(buffer).parent_path();
#else
    std::error_code ec;
    const std::filesystem::path exe = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (ec) return {};
    return exe.parent_path();
#endif
}

// Reads a SPIR-V binary. SPIR-V is a stream of 32-bit words, and
// vkCreateShaderModule requires 4-byte-aligned data, hence the uint32_t
// vector rather than a char vector.
std::vector<uint32_t> readSpirvFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("failed to open shader file: " + path.string());
    }
    const std::streamsize byteSize = file.tellg();
    if (byteSize <= 0 || byteSize % 4 != 0) {
        throw std::runtime_error("shader file is not valid SPIR-V: " + path.string());
    }
    std::vector<uint32_t> words(static_cast<size_t>(byteSize) / 4);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(words.data()), byteSize);
    if (!file) {
        throw std::runtime_error("failed to read shader file: " + path.string());
    }
    return words;
}

std::filesystem::path findShaderFile(const char* fileName) {
    const std::filesystem::path candidates[] = {
        executableDir() / "shaders" / fileName, // normal case: next to the exe
        std::filesystem::path("shaders") / fileName, // fallback: current dir
    };
    for (const std::filesystem::path& candidate : candidates) {
        std::error_code ec;
        if (std::filesystem::exists(candidate, ec)) return candidate;
    }
    throw std::runtime_error(std::string("compiled shader not found: ") + fileName +
                             " (looked next to the executable and in ./shaders)");
}

} // namespace

// ===========================================================================
// Construction / destruction.
// ===========================================================================

Renderer::Renderer(GLFWwindow* window, const Model& model,
                   const std::vector<InstanceData>& instances,
                   const std::vector<std::vector<uint8_t>>& textures)
    : window_(window) {
    // Strict bottom-up order; each stage depends only on the ones before it.
    // In WebGL2 the first four stages simply don't exist in application
    // code, and the rest were created lazily by the driver on first use.
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapchain();
    createSwapchainImageViews();
    createRenderPass();
    createDepthResources();
    createFramebuffers();
    createDescriptorSetLayout();
    createGraphicsPipeline();
    createCommandPool();
    createTextureArray(textures);
    createGeometryBuffers(model, instances);
    createUniformBuffers();
    createDescriptorSets();
    allocateCommandBuffers();
    createSyncObjects();
}

Renderer::~Renderer() {
    // Destruction order is the exact reverse of creation order. WebGL2 note:
    // in the browser all of this is garbage-collected when the context goes
    // away; in Vulkan destroying a parent object (the device) while a child
    // (a buffer) still exists is a leak the validation layers will name.
    if (device_ != VK_NULL_HANDLE) {
        // Nothing may be destroyed while the GPU could still be using it.
        vkDeviceWaitIdle(device_);

        cleanupSwapchain();

        for (uint32_t i = 0; i < kMaxFramesInFlight; ++i) {
            if (imageAvailable_[i] != VK_NULL_HANDLE) {
                vkDestroySemaphore(device_, imageAvailable_[i], nullptr);
            }
            if (inFlight_[i] != VK_NULL_HANDLE) {
                vkDestroyFence(device_, inFlight_[i], nullptr);
            }
        }

        // Command buffers are freed implicitly with their pool.
        if (commandPool_ != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device_, commandPool_, nullptr);
        }

        // Descriptor sets are freed implicitly with their pool.
        if (descriptorPool_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
        }
        if (descriptorSetLayout_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);
        }

        for (uint32_t i = 0; i < kMaxFramesInFlight; ++i) {
            // No vkUnmapMemory needed: freeing memory implicitly unmaps it.
            if (uniformBuffers_[i] != VK_NULL_HANDLE) {
                vkDestroyBuffer(device_, uniformBuffers_[i], nullptr);
            }
            if (uniformMemories_[i] != VK_NULL_HANDLE) {
                vkFreeMemory(device_, uniformMemories_[i], nullptr);
            }
        }

        if (textureSampler_ != VK_NULL_HANDLE) vkDestroySampler(device_, textureSampler_, nullptr);
        if (textureView_ != VK_NULL_HANDLE) vkDestroyImageView(device_, textureView_, nullptr);
        if (textureImage_ != VK_NULL_HANDLE) vkDestroyImage(device_, textureImage_, nullptr);
        if (textureMemory_ != VK_NULL_HANDLE) vkFreeMemory(device_, textureMemory_, nullptr);

        for (Chunk& chunk : chunks_) destroyChunk(chunk);
        for (Chunk& chunk : retiredChunks_) destroyChunk(chunk);

        if (instanceBuffer_ != VK_NULL_HANDLE) vkDestroyBuffer(device_, instanceBuffer_, nullptr);
        if (instanceMemory_ != VK_NULL_HANDLE) vkFreeMemory(device_, instanceMemory_, nullptr);
        if (indexBuffer_ != VK_NULL_HANDLE) vkDestroyBuffer(device_, indexBuffer_, nullptr);
        if (indexMemory_ != VK_NULL_HANDLE) vkFreeMemory(device_, indexMemory_, nullptr);
        if (vertexBuffer_ != VK_NULL_HANDLE) vkDestroyBuffer(device_, vertexBuffer_, nullptr);
        if (vertexMemory_ != VK_NULL_HANDLE) vkFreeMemory(device_, vertexMemory_, nullptr);

        if (pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, pipeline_, nullptr);
        if (pipelineLayout_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
        }
        if (renderPass_ != VK_NULL_HANDLE) vkDestroyRenderPass(device_, renderPass_, nullptr);

        vkDestroyDevice(device_, nullptr);
    }

    if (surface_ != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
    }
    if (debugMessenger_ != VK_NULL_HANDLE) {
        // Extension functions are not exported by the loader; they are
        // fetched by name at runtime (same story as eglGetProcAddress in GL
        // land - the web hid this behind the browser, native never did).
        auto destroyFn = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
        if (destroyFn != nullptr) destroyFn(instance_, debugMessenger_, nullptr);
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
    }
}

void Renderer::waitIdle() {
    if (device_ != VK_NULL_HANDLE) vkDeviceWaitIdle(device_);
}

// ===========================================================================
// Instance, debug messenger, surface.
// ===========================================================================

// The VkInstance is the connection between the application and the Vulkan
// loader - it exists *before* any GPU is involved. WebGL2 has no equivalent
// concept because the browser already decided which GPU and driver you get.
void Renderer::createInstance() {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "xrsps-native";
    appInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    appInfo.pEngineName = "xrsps";
    appInfo.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    // We target Vulkan 1.3 as the baseline. The renderer deliberately sticks
    // to the classic explicit objects (render passes, framebuffers) rather
    // than 1.3's dynamic rendering, because those objects *are* the lesson:
    // they make the structure of a frame visible and map directly onto what
    // tile-based mobile GPUs want to know up front.
    appInfo.apiVersion = VK_API_VERSION_1_3;

    // GLFW reports which surface extensions the platform needs
    // (VK_KHR_surface plus VK_KHR_win32_surface / VK_KHR_xcb_surface / ...).
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    if (glfwExtensions == nullptr) {
        throw std::runtime_error("GLFW reports no Vulkan support on this platform");
    }
    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    const bool useValidation = kEnableValidation && validationLayerAvailable();
    if (kEnableValidation && !useValidation) {
        std::fprintf(stderr,
                     "warning: %s not installed - running without validation\n",
                     kValidationLayerName);
    }
    if (useValidation) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    // The debug messenger proper is created *after* the instance, so to also
    // catch mistakes inside vkCreateInstance/vkDestroyInstance themselves the
    // spec allows chaining a messenger create-info into pNext here. pNext is
    // Vulkan's extension mechanism: a typed linked list of extra structs.
    VkDebugUtilsMessengerCreateInfoEXT debugInfo =
        makeDebugMessengerCreateInfo(&validationMessages_);
    if (useValidation) {
        createInfo.enabledLayerCount = 1;
        createInfo.ppEnabledLayerNames = &kValidationLayerName;
        createInfo.pNext = &debugInfo;
    }

    vkCheck(vkCreateInstance(&createInfo, nullptr, &instance_), "vkCreateInstance");
}

void Renderer::setupDebugMessenger() {
    if (!kEnableValidation || !validationLayerAvailable()) return;
    VkDebugUtilsMessengerCreateInfoEXT info =
        makeDebugMessengerCreateInfo(&validationMessages_);
    auto createFn = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
    if (createFn == nullptr) {
        throw std::runtime_error("vkCreateDebugUtilsMessengerEXT not available");
    }
    vkCheck(createFn(instance_, &info, nullptr, &debugMessenger_),
            "vkCreateDebugUtilsMessengerEXT");
}

// The VkSurfaceKHR ties Vulkan to a concrete window. In WebGL2 this binding
// is implicit - the context *is* the canvas. In Vulkan the surface is its
// own object because rendering does not require a window at all (headless
// compute, offscreen render farms); presentation is an extension, not a
// core feature. GLFW wraps the per-platform create call
// (vkCreateWin32SurfaceKHR, vkCreateXcbSurfaceKHR, ...) behind one portable
// function.
void Renderer::createSurface() {
    vkCheck(glfwCreateWindowSurface(instance_, window_, nullptr, &surface_),
            "glfwCreateWindowSurface");
}

// ===========================================================================
// Device selection and creation.
// ===========================================================================

// Enumerate every GPU the loader knows about and pick the most capable one
// that can render to our surface. WebGL2 note: the browser made this exact
// decision for you (usually via powerPreference), and you couldn't even
// find out what it picked without the WEBGL_debug_renderer_info extension.
void Renderer::pickPhysicalDevice() {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    if (count == 0) throw std::runtime_error("no Vulkan-capable GPUs found");
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance_, &count, devices.data());

    int bestScore = -1;
    for (VkPhysicalDevice candidate : devices) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(candidate, &props);

        // Hard requirements first.
        if (props.apiVersion < VK_API_VERSION_1_3) continue;
        if (!hasDeviceExtensions(candidate)) continue;
        if (findGraphicsPresentFamily(candidate, surface_) == UINT32_MAX) continue;

        uint32_t formatCount = 0;
        uint32_t presentModeCount = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(candidate, surface_, &formatCount, nullptr);
        vkGetPhysicalDeviceSurfacePresentModesKHR(candidate, surface_,
                                                  &presentModeCount, nullptr);
        if (formatCount == 0 || presentModeCount == 0) continue;

        // Preference order: discrete > integrated > virtual > software.
        int score = 0;
        switch (props.deviceType) {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: score = 400; break;
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: score = 300; break;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: score = 200; break;
            case VK_PHYSICAL_DEVICE_TYPE_CPU: score = 100; break;
            default: score = 50; break;
        }
        if (score > bestScore) {
            bestScore = score;
            physicalDevice_ = candidate;
            deviceProperties_ = props;
        }
    }
    if (physicalDevice_ == VK_NULL_HANDLE) {
        throw std::runtime_error(
            "no GPU supports Vulkan 1.3 + swapchain + graphics/present on this surface");
    }

    queueFamilyIndex_ = findGraphicsPresentFamily(physicalDevice_, surface_);

    // Pick the depth format up front (it depends only on the physical
    // device): prefer 32-bit float depth, fall back to the packed formats.
    // We never need stencil in this project.
    const VkFormat depthCandidates[] = {
        VK_FORMAT_D32_SFLOAT,
        VK_FORMAT_D32_SFLOAT_S8_UINT,
        VK_FORMAT_D24_UNORM_S8_UINT,
    };
    for (VkFormat format : depthCandidates) {
        VkFormatProperties props{};
        vkGetPhysicalDeviceFormatProperties(physicalDevice_, format, &props);
        if (props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
            depthFormat_ = format;
            break;
        }
    }
    if (depthFormat_ == VK_FORMAT_UNDEFINED) {
        throw std::runtime_error("no supported depth attachment format");
    }

    std::printf("GPU: %s (Vulkan %u.%u.%u)\n", deviceProperties_.deviceName,
                VK_API_VERSION_MAJOR(deviceProperties_.apiVersion),
                VK_API_VERSION_MINOR(deviceProperties_.apiVersion),
                VK_API_VERSION_PATCH(deviceProperties_.apiVersion));
}

// The logical device is our private connection to the chosen GPU, created
// with exactly the queues, extensions, and features we intend to use -
// Vulkan makes you declare all of it up front so the driver never has to
// guess (or lazily allocate) later.
void Renderer::createLogicalDevice() {
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = queueFamilyIndex_;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;

    // No optional GPU features needed: this renderer is deliberately within
    // core 1.0 functionality (plus the swapchain extension). If we used
    // e.g. samplerAnisotropy or wideLines, they would be switched on here.
    VkPhysicalDeviceFeatures features{};

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.queueCreateInfoCount = 1;
    createInfo.pQueueCreateInfos = &queueInfo;
    createInfo.enabledExtensionCount =
        static_cast<uint32_t>(std::size(kDeviceExtensions));
    createInfo.ppEnabledExtensionNames = kDeviceExtensions;
    createInfo.pEnabledFeatures = &features;
    // Note: device-level layers are deprecated; modern loaders use the
    // instance-level layers for everything, so none are listed here.

    vkCheck(vkCreateDevice(physicalDevice_, &createInfo, nullptr, &device_),
            "vkCreateDevice");

    // One queue does everything (graphics, presents, and staging transfers).
    // A production engine would also grab a dedicated DMA/transfer queue so
    // asset uploads overlap rendering.
    vkGetDeviceQueue(device_, queueFamilyIndex_, 0, &queue_);
}

// ===========================================================================
// Swapchain.
// ===========================================================================

// The swapchain is the set of images we take turns rendering into and
// handing to the OS compositor.
//
// WebGL2 -> Vulkan: this entire object is the "default framebuffer" you
// never saw. The browser owned the canvas backbuffer, chose its format,
// double/triple buffered it, and presented it after your requestAnimationFrame
// callback returned. Here every one of those policies is our decision, and
// the object becomes invalid (VK_ERROR_OUT_OF_DATE_KHR) whenever the window
// changes underneath it - see recreateSwapchain().
void Renderer::createSwapchain() {
    VkSurfaceCapabilitiesKHR caps{};
    vkCheck(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice_, surface_, &caps),
            "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");

    // --- format: prefer 8-bit BGRA with sRGB encoding-on-write, the same
    // behavior a WebGL2 canvas gave us implicitly.
    uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice_, surface_, &formatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice_, surface_, &formatCount,
                                         formats.data());
    VkSurfaceFormatKHR chosenFormat = formats[0];
    for (const VkSurfaceFormatKHR& format : formats) {
        if (format.format == VK_FORMAT_B8G8R8A8_SRGB &&
            format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            chosenFormat = format;
            break;
        }
    }

    // --- present mode: FIFO is the only mode the spec guarantees and is
    // classic vsync. We upgrade to MAILBOX when offered: it also never
    // tears, but instead of blocking, a newer frame replaces the queued one
    // (uncapped frame rate, lowest latency) - which is what we want for
    // reading the instancing cost off the frame-time display.
    VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
    uint32_t modeCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice_, surface_, &modeCount,
                                              nullptr);
    std::vector<VkPresentModeKHR> modes(modeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice_, surface_, &modeCount,
                                              modes.data());
    for (VkPresentModeKHR mode : modes) {
        if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {
            presentMode = mode;
            break;
        }
    }

    // --- extent: normally the surface dictates its size; a special value
    // of 0xFFFFFFFF means "you choose", in which case we clamp the current
    // framebuffer size (in pixels, NOT screen coordinates - they differ on
    // high-DPI displays) to the allowed range.
    if (caps.currentExtent.width != 0xFFFFFFFFu) {
        swapchainExtent_ = caps.currentExtent;
    } else {
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window_, &width, &height);
        swapchainExtent_.width =
            std::clamp(static_cast<uint32_t>(width), caps.minImageExtent.width,
                       caps.maxImageExtent.width);
        swapchainExtent_.height =
            std::clamp(static_cast<uint32_t>(height), caps.minImageExtent.height,
                       caps.maxImageExtent.height);
    }

    // --- image count: the minimum plus one, so the driver handing an image
    // to the compositor never leaves us with zero images to acquire.
    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0) imageCount = std::min(imageCount, caps.maxImageCount);

    // --- usage: render target, plus transfer-source so the screenshot path
    // can copy the presented image back to the CPU. TRANSFER_SRC is
    // near-universally supported on swapchains, but it is not guaranteed,
    // so screenshots degrade gracefully instead of failing creation.
    VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    supportsScreenshots_ =
        (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0;
    if (supportsScreenshots_) usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    // --- composite alpha: we don't blend with other windows; prefer OPAQUE
    // but take whatever single mode the platform offers (Wayland surfaces,
    // for example, often only advertise pre-multiplied or inherit).
    VkCompositeAlphaFlagBitsKHR compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    if ((caps.supportedCompositeAlpha & compositeAlpha) == 0) {
        compositeAlpha = static_cast<VkCompositeAlphaFlagBitsKHR>(
            1u << std::countr_zero(caps.supportedCompositeAlpha));
    }

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface_;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = chosenFormat.format;
    createInfo.imageColorSpace = chosenFormat.colorSpace;
    createInfo.imageExtent = swapchainExtent_;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = usage;
    // One queue family touches these images, so EXCLUSIVE sharing (the fast
    // path). CONCURRENT would be needed with split graphics/present families.
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.preTransform = caps.currentTransform;
    createInfo.compositeAlpha = compositeAlpha;
    createInfo.presentMode = presentMode;
    // Allow the driver to skip shading pixels hidden behind other windows.
    createInfo.clipped = VK_TRUE;
    // On resize we fully tear down and rebuild (simpler to reason about).
    // Production note: passing the old swapchain here lets the driver
    // recycle images and avoids a visible hitch on resize.
    createInfo.oldSwapchain = VK_NULL_HANDLE;

    vkCheck(vkCreateSwapchainKHR(device_, &createInfo, nullptr, &swapchain_),
            "vkCreateSwapchainKHR");
    swapchainFormat_ = chosenFormat.format;

    // The driver may create MORE images than we asked for; always re-query.
    uint32_t actualCount = 0;
    vkGetSwapchainImagesKHR(device_, swapchain_, &actualCount, nullptr);
    swapchainImages_.resize(actualCount);
    vkGetSwapchainImagesKHR(device_, swapchain_, &actualCount, swapchainImages_.data());
}

// Raw VkImages can't be used as attachments directly; a VkImageView states
// how the image will be interpreted (2D color, which mip/array range).
// WebGL2 blurred image and view into one WebGLTexture object.
void Renderer::createSwapchainImageViews() {
    swapchainImageViews_.resize(swapchainImages_.size());
    for (size_t i = 0; i < swapchainImages_.size(); ++i) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = swapchainImages_[i];
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = swapchainFormat_;
        viewInfo.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                               VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
        viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCheck(vkCreateImageView(device_, &viewInfo, nullptr, &swapchainImageViews_[i]),
                "vkCreateImageView(swapchain)");
    }
}

// ===========================================================================
// Render pass, depth buffer, framebuffers.
// ===========================================================================

// A render pass declares, ahead of time, every attachment a chunk of
// rendering will touch, what to do with it at the start (clear? keep?) and
// end (store? discard?), and which layout each image must be in.
//
// WebGL2 -> Vulkan: WebGL2 had no such object - you set up an FBO and drew.
// The declaration exists because it's exactly the information a tile-based
// GPU needs to keep attachments in on-chip tile memory: "CLEAR + STORE"
// means "no need to read the old contents from RAM"; the depth buffer's
// STORE_OP_DONT_CARE below means "never write depth back to RAM at all".
// On desktop it's also where image layout transitions are scheduled for us.
void Renderer::createRenderPass() {
    VkAttachmentDescription color{};
    color.format = swapchainFormat_;
    color.samples = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    // UNDEFINED tells Vulkan we don't care what was in the image before -
    // the driver may skip restoring it. The pass itself transitions the
    // image to PRESENT_SRC when it ends; no manual barrier needed.
    color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentDescription depth{};
    depth.format = depthFormat_;
    depth.samples = VK_SAMPLE_COUNT_1_BIT;
    depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    // The depth buffer only matters DURING the frame; nothing reads it
    // afterwards, so the GPU may throw it away instead of writing it out.
    depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depth.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    subpass.pDepthStencilAttachment = &depthRef;

    // This dependency orders our pass against whatever used the attachments
    // before it (EXTERNAL = "commands earlier in the queue"). Two frames in
    // flight share ONE depth image; this is what stops frame N+1's depth
    // clear from racing frame N's depth writes - and on the color side it
    // orders the pass after the presentation engine's read of the image
    // (chained through the imageAvailable semaphore wait, which happens-in
    // COLOR_ATTACHMENT_OUTPUT stage).
    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                              VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                               VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                              VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                               VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    VkAttachmentDescription attachments[] = {color, depth};
    VkRenderPassCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    createInfo.attachmentCount = 2;
    createInfo.pAttachments = attachments;
    createInfo.subpassCount = 1;
    createInfo.pSubpasses = &subpass;
    createInfo.dependencyCount = 1;
    createInfo.pDependencies = &dependency;

    vkCheck(vkCreateRenderPass(device_, &createInfo, nullptr, &renderPass_),
            "vkCreateRenderPass");
}

// The depth buffer. WebGL2 note: `getContext("webgl2", {depth: true})` was
// the entire WebGL2 version of this function. Here we allocate the image,
// pick its memory type, bind it, and create a view - the same three-step
// dance as any texture, because a depth buffer IS just an image with a
// depth format. One depth image is shared by both frames in flight; the
// render pass dependency above serializes access to it.
void Renderer::createDepthResources() {
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = depthFormat_;
    imageInfo.extent = {swapchainExtent_.width, swapchainExtent_.height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    // OPTIMAL = driver-private tiled layout (fast); LINEAR = row-major
    // (readable by the CPU, slow as an attachment). Always OPTIMAL for
    // anything the GPU renders into.
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    vkCheck(vkCreateImage(device_, &imageInfo, nullptr, &depthImage_),
            "vkCreateImage(depth)");

    // Explicit memory management, the part WebGL2 never showed you: ask what
    // the image needs, find a device-local heap that satisfies it, allocate,
    // bind. Production note: real engines allocate a few big VkDeviceMemory
    // blocks and sub-allocate (usually via VMA), because implementations may
    // cap total allocations at 4096; one-allocation-per-resource is only
    // acceptable in a sample this size.
    VkMemoryRequirements memReq{};
    vkGetImageMemoryRequirements(device_, depthImage_, &memReq);
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex =
        findMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkCheck(vkAllocateMemory(device_, &allocInfo, nullptr, &depthMemory_),
            "vkAllocateMemory(depth)");
    vkCheck(vkBindImageMemory(device_, depthImage_, depthMemory_, 0),
            "vkBindImageMemory(depth)");

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = depthImage_;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = depthFormat_;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    vkCheck(vkCreateImageView(device_, &viewInfo, nullptr, &depthView_),
            "vkCreateImageView(depth)");

    // No manual layout transition needed: the render pass declares
    // initialLayout UNDEFINED -> DEPTH_STENCIL_ATTACHMENT_OPTIMAL and the
    // driver folds the transition into the pass.
}

// A framebuffer binds concrete image views into the render pass's attachment
// slots - one per swapchain image, all sharing the single depth buffer.
// Closest WebGL2 concept: a completed FBO, except compatibility with the
// pass is validated once at creation, not re-checked per draw.
void Renderer::createFramebuffers() {
    framebuffers_.resize(swapchainImageViews_.size());
    for (size_t i = 0; i < swapchainImageViews_.size(); ++i) {
        const VkImageView attachments[] = {swapchainImageViews_[i], depthView_};
        VkFramebufferCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        createInfo.renderPass = renderPass_;
        createInfo.attachmentCount = 2;
        createInfo.pAttachments = attachments;
        createInfo.width = swapchainExtent_.width;
        createInfo.height = swapchainExtent_.height;
        createInfo.layers = 1;
        vkCheck(vkCreateFramebuffer(device_, &createInfo, nullptr, &framebuffers_[i]),
                "vkCreateFramebuffer");
    }
}

// ===========================================================================
// Descriptors and pipeline.
// ===========================================================================

// The descriptor set LAYOUT is the shader's resource signature: "set 0,
// binding 0 is a uniform buffer visible to the vertex stage". It's part of
// the pipeline's interface contract, separate from the actual buffers
// (bound later via descriptor SETS).
//
// WebGL2 -> Vulkan: WebGL2 uniforms were named slots queried with
// getUniformLocation and set one at a time; uniform *blocks* got halfway
// here. Vulkan drops names entirely - binding numbers in the shader and
// this layout are the whole contract.
void Renderer::createDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding bindings[2]{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    // The world-texture array; sampled in the fragment stage by faces whose
    // vertices carry a texture layer.
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    createInfo.bindingCount = 2;
    createInfo.pBindings = bindings;
    vkCheck(vkCreateDescriptorSetLayout(device_, &createInfo, nullptr,
                                        &descriptorSetLayout_),
            "vkCreateDescriptorSetLayout");
}

VkShaderModule Renderer::loadShaderModule(const char* fileName) {
    const std::vector<uint32_t> code = readSpirvFile(findShaderFile(fileName));
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size() * sizeof(uint32_t);
    createInfo.pCode = code.data();
    VkShaderModule shaderModule = VK_NULL_HANDLE;
    vkCheck(vkCreateShaderModule(device_, &createInfo, nullptr, &shaderModule),
            "vkCreateShaderModule");
    return shaderModule;
}

// The graphics pipeline object bakes nearly ALL rendering state at creation
// time: shaders, vertex layout, rasterizer mode, depth rules, blending.
//
// WebGL2 -> Vulkan, the big one: WebGL2 was a state machine - each of these
// was a separate gl* call you could flip between draws (gl.enable(DEPTH_TEST),
// gl.cullFace, gl.useProgram, ...), and the driver secretly recompiled
// pipeline variants behind your back when combinations changed (that's what
// caused mysterious first-use hitches). Vulkan makes the full combination
// one immutable object, compiled once, here, at a time we control. The cost:
// N distinct state combinations = N pipeline objects, created up front.
void Renderer::createGraphicsPipeline() {
    VkShaderModule vertModule = loadShaderModule("model.vert.spv");
    VkShaderModule fragModule = loadShaderModule("model.frag.spv");

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertModule;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragModule;
    stages[1].pName = "main";

    // Two vertex buffer bindings with different step rates:
    //   binding 0 advances per VERTEX (positions/normals/colors),
    //   binding 1 advances per INSTANCE (a mat4 world transform).
    // WebGL2 expressed the same thing with vertexAttribPointer +
    // vertexAttribDivisor(loc, 1); Vulkan bakes it into the pipeline.
    VkVertexInputBindingDescription bindings[2]{};
    bindings[0].binding = 0;
    bindings[0].stride = sizeof(Vertex);
    bindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    bindings[1].binding = 1;
    bindings[1].stride = sizeof(InstanceData);
    bindings[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

    // A mat4 attribute is described as four vec4 columns in consecutive
    // locations (4..7) - there is no "mat4" vertex format on any API.
    VkVertexInputAttributeDescription attributes[8]{};
    attributes[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)};
    attributes[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)};
    attributes[2] = {2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color)};
    attributes[3] = {3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, uvLayer)};
    for (uint32_t column = 0; column < 4; ++column) {
        attributes[4 + column] = {4 + column, 1, VK_FORMAT_R32G32B32A32_SFLOAT,
                                  static_cast<uint32_t>(sizeof(glm::vec4)) * column};
    }

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 2;
    vertexInput.pVertexBindingDescriptions = bindings;
    vertexInput.vertexAttributeDescriptionCount = 8;
    vertexInput.pVertexAttributeDescriptions = attributes;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    // Design decision: viewport and scissor are DYNAMIC state - set per
    // command buffer, not baked here. Baking them is marginally faster on
    // some drivers, but would force a pipeline rebuild on every window
    // resize; dynamic viewport/scissor costs nothing measurable and is what
    // real engines do. (This tiny exception proves the "bake everything"
    // rule: Vulkan lets you opt individual states back out.)
    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_BACK_BIT;
    // The model is authored counter-clockwise viewed from outside (the GL
    // convention), and CCW is also what the rasterizer sees - via a double
    // negation worth spelling out. The projection Y flip (camera.cpp)
    // mirrors the winding once; but Vulkan evaluates winding in framebuffer
    // space, whose Y axis points DOWN (GL's pointed up), which mirrors the
    // reading back. The two cancel, so GL-convention geometry with the
    // GL-style Y-flipped projection keeps its GL front face. Getting this
    // wrong doesn't crash anything - you just silently render the interior
    // of every mesh (back faces), the classic Vulkan-port bug.
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // Standard less-than depth testing. In WebGL2: gl.enable(gl.DEPTH_TEST)
    // + gl.depthFunc(gl.LESS), mutable global state; here, immutable
    // pipeline state.
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

    // Opaque geometry only: blending disabled, all channels written.
    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo blend{};
    blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = 1;
    blend.pAttachments = &blendAttachment;

    const VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT,
                                            VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{};
    dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamicStates;

    // The pipeline layout is the full resource interface: descriptor sets
    // plus push-constant ranges (none here - the per-frame matrices ride in
    // a UBO; push constants would be the alternative for tiny data).
    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &descriptorSetLayout_;
    vkCheck(vkCreatePipelineLayout(device_, &layoutInfo, nullptr, &pipelineLayout_),
            "vkCreatePipelineLayout");

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = stages;
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &raster;
    pipelineInfo.pMultisampleState = &multisample;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &blend;
    pipelineInfo.pDynamicState = &dynamic;
    pipelineInfo.layout = pipelineLayout_;
    pipelineInfo.renderPass = renderPass_;
    pipelineInfo.subpass = 0;

    // Production note: the VK_NULL_HANDLE here is a VkPipelineCache - a real
    // engine passes one (persisted to disk) so pipeline compilation survives
    // restarts, and compiles pipelines on worker threads.
    vkCheck(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo,
                                      nullptr, &pipeline_),
            "vkCreateGraphicsPipelines");

    // Modules are just SPIR-V containers; once the pipeline has consumed
    // them they can go.
    vkDestroyShaderModule(device_, fragModule, nullptr);
    vkDestroyShaderModule(device_, vertModule, nullptr);
}

// ===========================================================================
// Command pool, geometry upload, uniform buffers, descriptor sets.
// ===========================================================================

void Renderer::createCommandPool() {
    VkCommandPoolCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    // We re-record each frame's command buffer every frame, so allow
    // individual resets. The alternative (resetting the whole pool) is what
    // multi-threaded engines do with one pool per thread per frame.
    createInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    createInfo.queueFamilyIndex = queueFamilyIndex_;
    vkCheck(vkCreateCommandPool(device_, &createInfo, nullptr, &commandPool_),
            "vkCreateCommandPool");
}

// Uploads the game textures (128x128 RGBA each) as one 2D array image and
// creates the sampler. Same staging discipline as the vertex buffers, plus
// the image-specific extras: explicit layout transitions on either side of
// the copy, and a view + sampler pair for the fragment stage.
void Renderer::createTextureArray(const std::vector<std::vector<uint8_t>>& textures) {
    constexpr uint32_t kTextureSize = 128;
    constexpr VkDeviceSize kLayerBytes = VkDeviceSize{kTextureSize} * kTextureSize * 4;

    // Descriptors must point at something even in untextured scenes, so an
    // empty input becomes a single opaque white layer that nothing samples.
    const uint32_t layerCount = std::max<size_t>(textures.size(), 1);

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
    imageInfo.extent = {kTextureSize, kTextureSize, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = layerCount;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    vkCheck(vkCreateImage(device_, &imageInfo, nullptr, &textureImage_),
            "vkCreateImage(textures)");

    VkMemoryRequirements memReq{};
    vkGetImageMemoryRequirements(device_, textureImage_, &memReq);
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex =
        findMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkCheck(vkAllocateMemory(device_, &allocInfo, nullptr, &textureMemory_),
            "vkAllocateMemory(textures)");
    vkCheck(vkBindImageMemory(device_, textureImage_, textureMemory_, 0),
            "vkBindImageMemory(textures)");

    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    createBuffer(kLayerBytes * layerCount, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 staging, stagingMemory);
    void* mapped = nullptr;
    vkCheck(vkMapMemory(device_, stagingMemory, 0, kLayerBytes * layerCount, 0, &mapped),
            "vkMapMemory(textures)");
    if (textures.empty()) {
        std::memset(mapped, 0xff, static_cast<size_t>(kLayerBytes));
    } else {
        for (size_t i = 0; i < textures.size(); ++i) {
            std::memcpy(static_cast<uint8_t*>(mapped) + kLayerBytes * i,
                        textures[i].data(),
                        std::min<size_t>(textures[i].size(), kLayerBytes));
        }
    }
    vkUnmapMemory(device_, stagingMemory);

    VkCommandBuffer cmd = beginOneShotCommands();
    VkImageMemoryBarrier toDst{};
    toDst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toDst.srcAccessMask = 0;
    toDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toDst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    toDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toDst.image = textureImage_;
    toDst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layerCount};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                         &toDst);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, layerCount};
    region.imageExtent = {kTextureSize, kTextureSize, 1};
    vkCmdCopyBufferToImage(cmd, staging, textureImage_,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    VkImageMemoryBarrier toRead = toDst;
    toRead.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toRead.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    toRead.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toRead.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &toRead);
    endOneShotCommands(cmd);

    vkDestroyBuffer(device_, staging, nullptr);
    vkFreeMemory(device_, stagingMemory, nullptr);

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = textureImage_;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    viewInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layerCount};
    vkCheck(vkCreateImageView(device_, &viewInfo, nullptr, &textureView_),
            "vkCreateImageView(textures)");

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.maxLod = 0.25f; // single mip; production would generate a chain
    vkCheck(vkCreateSampler(device_, &samplerInfo, nullptr, &textureSampler_),
            "vkCreateSampler");
}

// Uploads the model exactly once. Geometry lives in DEVICE_LOCAL memory
// (VRAM on a discrete GPU), which the CPU cannot write on most hardware, so
// each buffer takes the staging route: CPU -> host-visible staging buffer ->
// GPU copy command -> device-local buffer.
//
// WebGL2 -> Vulkan: all three uploads together replace three calls to
// gl.bufferData(target, data, STATIC_DRAW). The "STATIC_DRAW" hint was the
// browser's version of this decision - it *probably* staged into VRAM,
// eventually, on its own schedule. Here the copy is explicit, and we know
// exactly when it completed (endOneShotCommands blocks on a fence).
void Renderer::createGeometryBuffers(const Model& model,
                                     const std::vector<InstanceData>& instances) {
    if (instances.empty()) {
        throw std::runtime_error("instance data must not be empty");
    }
    indexCount_ = static_cast<uint32_t>(model.indices.size());
    instanceCapacity_ = static_cast<uint32_t>(instances.size());

    // An empty base model is legal: world mode streams all of its geometry
    // in as chunks at runtime and starts with nothing resident.
    if (indexCount_ > 0) {
        createDeviceLocalBuffer(model.vertices.data(),
                                model.vertices.size() * sizeof(Vertex),
                                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexBuffer_,
                                vertexMemory_);
        createDeviceLocalBuffer(model.indices.data(),
                                model.indices.size() * sizeof(uint32_t),
                                VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indexBuffer_,
                                indexMemory_);
    }
    // The instance transforms are static for the app's lifetime, which is
    // the point of the instancing demo: per-frame cost is one draw call
    // whose instance count changes, not N transform uploads.
    createDeviceLocalBuffer(instances.data(), instances.size() * sizeof(InstanceData),
                            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, instanceBuffer_,
                            instanceMemory_);

    std::printf("Model: %u vertices, %u triangles | %u instance transforms\n",
                static_cast<uint32_t>(model.vertices.size()), indexCount_ / 3,
                instanceCapacity_);
}

// One uniform buffer per frame in flight, persistently mapped.
//
// Why per frame in flight: frame N's UBO may still be feeding the GPU while
// the CPU records frame N+1. One buffer would be a data race; two lets each
// frame own its slot (the fence in drawFrame guarantees slot N is idle
// before we overwrite it).
//
// Why persistently mapped HOST_VISIBLE|HOST_COHERENT rather than
// device-local + staging: the block is 160 bytes, rewritten every frame;
// a staging copy would cost more than the GPU reading it across the bus.
// WebGL2 note: this replaces per-draw gl.uniform*/bufferSubData calls, and
// "map once, keep the pointer" has no WebGL2 equivalent at all - the
// browser always copied.
void Renderer::createUniformBuffers() {
    for (uint32_t i = 0; i < kMaxFramesInFlight; ++i) {
        createBuffer(sizeof(FrameUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     uniformBuffers_[i], uniformMemories_[i]);
        vkCheck(vkMapMemory(device_, uniformMemories_[i], 0, sizeof(FrameUniforms), 0,
                            &uniformMapped_[i]),
                "vkMapMemory(uniform)");
    }
}

// Descriptor sets are the actual binding of "buffer X" into "slot 0" - the
// layout said what kind of thing goes in the slot, the set says which one.
// Sets are allocated from a pool (sized exactly for our needs) and written
// once; they never change afterwards, so there is no per-frame descriptor
// churn in this app.
void Renderer::createDescriptorSets() {
    VkDescriptorPoolSize poolSizes[2]{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = kMaxFramesInFlight;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = kMaxFramesInFlight;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = kMaxFramesInFlight;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = poolSizes;
    vkCheck(vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPool_),
            "vkCreateDescriptorPool");

    VkDescriptorSetLayout layouts[kMaxFramesInFlight];
    for (uint32_t i = 0; i < kMaxFramesInFlight; ++i) layouts[i] = descriptorSetLayout_;

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool_;
    allocInfo.descriptorSetCount = kMaxFramesInFlight;
    allocInfo.pSetLayouts = layouts;
    vkCheck(vkAllocateDescriptorSets(device_, &allocInfo, descriptorSets_),
            "vkAllocateDescriptorSets");

    for (uint32_t i = 0; i < kMaxFramesInFlight; ++i) {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = uniformBuffers_[i];
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(FrameUniforms);

        VkDescriptorImageInfo imageInfo{};
        imageInfo.sampler = textureSampler_;
        imageInfo.imageView = textureView_;
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet writes[2]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = descriptorSets_[i];
        writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].descriptorCount = 1;
        writes[0].pBufferInfo = &bufferInfo;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = descriptorSets_[i];
        writes[1].dstBinding = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].descriptorCount = 1;
        writes[1].pImageInfo = &imageInfo;
        vkUpdateDescriptorSets(device_, 2, writes, 0, nullptr);
    }
}

void Renderer::allocateCommandBuffers() {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool_;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = kMaxFramesInFlight;
    vkCheck(vkAllocateCommandBuffers(device_, &allocInfo, commandBuffers_),
            "vkAllocateCommandBuffers");
}

// ===========================================================================
// Synchronization objects.
//
// WebGL2 -> Vulkan, the deep end: WebGL2 rendering LOOKED synchronous -
// gl.drawArrays "just worked" and the browser presented after your rAF
// callback returned. In reality the driver pipelined everything and stalled
// you whenever a hazard appeared. Vulkan hands us the actual moving parts:
//
//   - semaphore  = GPU->GPU ordering (queue waits for queue/presentation
//                  engine; the CPU never observes it)
//   - fence      = GPU->CPU ordering (the CPU blocks on / polls it)
//
// Two frames in flight means the CPU may record frame N+1 while the GPU
// still executes frame N - that overlap is the entire performance win of
// this design. Per-frame resources that the CPU rewrites (command buffer,
// uniform slot) exist twice; the fence is what proves slot reuse is safe.
// ===========================================================================
void Renderer::createSyncObjects() {
    VkSemaphoreCreateInfo semInfo{};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    // Created signaled so drawFrame's very first wait doesn't deadlock.
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (uint32_t i = 0; i < kMaxFramesInFlight; ++i) {
        vkCheck(vkCreateSemaphore(device_, &semInfo, nullptr, &imageAvailable_[i]),
                "vkCreateSemaphore(imageAvailable)");
        vkCheck(vkCreateFence(device_, &fenceInfo, nullptr, &inFlight_[i]),
                "vkCreateFence");
    }
    createRenderFinishedSemaphores();
}

// The renderFinished semaphores are sized per SWAPCHAIN IMAGE, not per frame
// in flight, and that distinction is load-bearing: vkQueuePresentKHR gives
// us no fence, so there is no way to know when a *frame-owned* semaphore is
// done being waited on by the presentation engine - reusing one two frames
// later could re-signal it while still in use. Keying them by image index
// is safe because acquiring that image again proves its previous present
// completed. (Validation layers newer than ~1.3.28x flag the per-frame
// variant, which most older tutorials still teach.)
void Renderer::createRenderFinishedSemaphores() {
    VkSemaphoreCreateInfo semInfo{};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    renderFinished_.resize(swapchainImages_.size());
    for (VkSemaphore& semaphore : renderFinished_) {
        vkCheck(vkCreateSemaphore(device_, &semInfo, nullptr, &semaphore),
                "vkCreateSemaphore(renderFinished)");
    }
}

// ===========================================================================
// Swapchain lifecycle (resize / minimize).
// ===========================================================================

// Destroys everything whose lifetime is tied to the swapchain: the
// framebuffers and depth buffer match its extent, the renderFinished
// semaphores match its image count. The render pass and pipeline survive -
// they depend only on formats, which are stable for a given surface, and
// viewport/scissor are dynamic state precisely so resizing does not touch
// the pipeline.
void Renderer::cleanupSwapchain() {
    for (VkFramebuffer framebuffer : framebuffers_) {
        vkDestroyFramebuffer(device_, framebuffer, nullptr);
    }
    framebuffers_.clear();

    if (depthView_ != VK_NULL_HANDLE) vkDestroyImageView(device_, depthView_, nullptr);
    if (depthImage_ != VK_NULL_HANDLE) vkDestroyImage(device_, depthImage_, nullptr);
    if (depthMemory_ != VK_NULL_HANDLE) vkFreeMemory(device_, depthMemory_, nullptr);
    depthView_ = VK_NULL_HANDLE;
    depthImage_ = VK_NULL_HANDLE;
    depthMemory_ = VK_NULL_HANDLE;

    for (VkSemaphore semaphore : renderFinished_) {
        vkDestroySemaphore(device_, semaphore, nullptr);
    }
    renderFinished_.clear();

    for (VkImageView view : swapchainImageViews_) {
        vkDestroyImageView(device_, view, nullptr);
    }
    swapchainImageViews_.clear();

    if (swapchain_ != VK_NULL_HANDLE) vkDestroySwapchainKHR(device_, swapchain_, nullptr);
    swapchain_ = VK_NULL_HANDLE;
}

// WebGL2 note: canvas resize was one line (canvas.width = w) and the browser
// rebuilt the backbuffer for you. This function is that line, expanded.
void Renderer::recreateSwapchain() {
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window_, &width, &height);
    if (width == 0 || height == 0) {
        // Minimized. drawFrame skips rendering while the framebuffer is
        // 0x0, so simply keep the old (stale) swapchain until we are
        // visible again; the next drawFrame will land back here.
        return;
    }

    // Brute-force but airtight: wait for the GPU to go idle, rebuild.
    // Production note: a shipping renderer would instead pass the old
    // swapchain to vkCreateSwapchainKHR and keep rendering during the
    // handover - full idle stalls every queue.
    vkDeviceWaitIdle(device_);
    cleanupSwapchain();
    createSwapchain();
    createSwapchainImageViews();
    createDepthResources();
    createFramebuffers();
    createRenderFinishedSemaphores();
}

// ===========================================================================
// Per-frame path.
// ===========================================================================

void Renderer::updateUniformBuffer(uint32_t frameIndex, const FrameInput& input) {
    FrameUniforms uniforms{};
    uniforms.view = input.view;
    uniforms.proj = input.proj;
    uniforms.lightDirection = glm::vec4(glm::normalize(kLightDirection), 0.0f);
    uniforms.lightParams = glm::vec4(input.ambient, input.diffuse, 0.0f, 0.0f);
    // Plain memcpy through the persistently mapped pointer. Safe because
    // drawFrame already waited on this frame slot's fence, proving the GPU
    // finished reading this buffer's previous contents; coherent memory
    // makes the write visible without an explicit flush.
    std::memcpy(uniformMapped_[frameIndex], &uniforms, sizeof(uniforms));
}

// Records the entire frame into `cmd`. WebGL2 -> Vulkan: in WebGL2 each
// gl.draw* was dispatched immediately (well - it *looked* that way; the
// browser actually batched behind your back). Vulkan makes the batching
// explicit: nothing here executes now; we are writing a list that
// vkQueueSubmit hands to the GPU as one unit. This is also what makes
// multi-threaded rendering possible - command buffers can be recorded on
// any thread, in parallel (a production note, not done here).
void Renderer::recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex,
                                   const FrameInput& input, bool captureAfterRender) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkCheck(vkBeginCommandBuffer(cmd, &beginInfo), "vkBeginCommandBuffer");

    VkClearValue clearValues[2]{};
    clearValues[0].color = kClearColor;
    // Depth clears to 1.0 = the far plane, with COMPARE_OP_LESS bringing
    // anything nearer forward. (Production note: many engines flip to
    // "reversed-Z" - clear to 0, GREATER compare, float depth - for hugely
    // better precision distribution; not needed at this scene scale.)
    clearValues[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo passInfo{};
    passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    passInfo.renderPass = renderPass_;
    passInfo.framebuffer = framebuffers_[imageIndex];
    passInfo.renderArea = {{0, 0}, swapchainExtent_};
    passInfo.clearValueCount = 2;
    passInfo.pClearValues = clearValues;

    vkCmdBeginRenderPass(cmd, &passInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);

    // Dynamic state promised by the pipeline; must be set before drawing.
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(swapchainExtent_.width);
    viewport.height = static_cast<float>(swapchainExtent_.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    VkRect2D scissor{{0, 0}, swapchainExtent_};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout_, 0, 1,
                            &descriptorSets_[currentFrame_], 0, nullptr);

    const VkDeviceSize offsets[] = {0, 0};
    if (indexCount_ > 0) {
        const VkBuffer vertexBuffers[] = {vertexBuffer_, instanceBuffer_};
        vkCmdBindVertexBuffers(cmd, 0, 2, vertexBuffers, offsets);
        vkCmdBindIndexBuffer(cmd, indexBuffer_, 0, VK_INDEX_TYPE_UINT32);

        // Defensive clamp: drawing past the end of the instance buffer is
        // undefined behavior no layer can reliably catch (the GPU would
        // fetch garbage transforms).
        const uint32_t firstInstance = std::min(input.firstInstance, instanceCapacity_ - 1);
        const uint32_t instanceCount =
            std::min(input.instanceCount, instanceCapacity_ - firstInstance);

        // The one draw call. Whether this renders 1 tree or the whole N x N
        // grid, the CPU cost is identical - the instance count is just a
        // number in the command; the GPU steps binding 1 once per instance.
        // That is the entire thesis of instancing: per-frame cost scales
        // with unique geometry, not with how many copies of it you draw.
        vkCmdDrawIndexed(cmd, indexCount_, instanceCount, 0, 0, firstInstance);
    }

    // Streamed world chunks: same pipeline and descriptors, one identity
    // instance each; only the vertex/index bindings change per chunk.
    for (const Chunk& chunk : chunks_) {
        const VkBuffer vertexBuffers[] = {chunk.vertexBuffer, instanceBuffer_};
        vkCmdBindVertexBuffers(cmd, 0, 2, vertexBuffers, offsets);
        vkCmdBindIndexBuffer(cmd, chunk.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, chunk.indexCount, 1, 0, 0, 0);
    }

    vkCmdEndRenderPass(cmd);

    if (captureAfterRender) {
        // The render pass just transitioned the image to PRESENT_SRC. Slip a
        // copy in before presentation: PRESENT_SRC -> TRANSFER_SRC, copy the
        // pixels into a host-visible buffer, and back. An explicit pipeline
        // barrier both orders the copy after the color writes AND performs
        // the layout transitions - in WebGL2, gl.readPixels did all of this
        // invisibly (and stalled the whole pipeline doing it).
        VkImageMemoryBarrier toTransfer{};
        toTransfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toTransfer.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        toTransfer.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toTransfer.image = swapchainImages_[imageIndex];
        toTransfer.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                             1, &toTransfer);

        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {swapchainExtent_.width, swapchainExtent_.height, 1};
        vkCmdCopyImageToBuffer(cmd, swapchainImages_[imageIndex],
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, screenshotBuffer_,
                               1, &region);

        VkImageMemoryBarrier toPresent = toTransfer;
        toPresent.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        toPresent.dstAccessMask = 0; // presentation performs its own visibility
        toPresent.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        toPresent.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0,
                             nullptr, 1, &toPresent);
    }

    vkCheck(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
}

uint64_t Renderer::addChunkGeometry(const Model& model) {
    if (model.vertices.empty() || model.indices.empty()) return 0;
    Chunk chunk;
    chunk.id = nextChunkId_++;
    chunk.indexCount = static_cast<uint32_t>(model.indices.size());
    createDeviceLocalBuffer(model.vertices.data(), model.vertices.size() * sizeof(Vertex),
                            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, chunk.vertexBuffer,
                            chunk.vertexMemory);
    createDeviceLocalBuffer(model.indices.data(), model.indices.size() * sizeof(uint32_t),
                            VK_BUFFER_USAGE_INDEX_BUFFER_BIT, chunk.indexBuffer,
                            chunk.indexMemory);
    chunks_.push_back(chunk);
    return chunk.id;
}

void Renderer::removeChunkGeometry(uint64_t handle) {
    if (handle == 0) return;
    for (size_t i = 0; i < chunks_.size(); ++i) {
        if (chunks_[i].id == handle) {
            chunks_[i].retiredAtFrame = absoluteFrame_;
            retiredChunks_.push_back(chunks_[i]);
            chunks_.erase(chunks_.begin() + static_cast<ptrdiff_t>(i));
            return;
        }
    }
}

void Renderer::destroyChunk(Chunk& chunk) {
    vkDestroyBuffer(device_, chunk.vertexBuffer, nullptr);
    vkFreeMemory(device_, chunk.vertexMemory, nullptr);
    vkDestroyBuffer(device_, chunk.indexBuffer, nullptr);
    vkFreeMemory(device_, chunk.indexMemory, nullptr);
}

bool Renderer::drawFrame(const FrameInput& input) {
    // Minimized windows have a 0x0 framebuffer, which is not a legal
    // swapchain extent - skip the frame entirely. (Design decision: skipping
    // keeps the main loop responsive for scripted tests; the alternative,
    // blocking in glfwWaitEvents until restored, burns less CPU and suits a
    // real app.)
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window_, &width, &height);
    if (width == 0 || height == 0) return false;

    // Free retired chunk buffers once every frame that could have recorded
    // them has certainly finished on the GPU.
    ++absoluteFrame_;
    for (size_t i = retiredChunks_.size(); i-- > 0;) {
        if (absoluteFrame_ - retiredChunks_[i].retiredAtFrame > kMaxFramesInFlight) {
            destroyChunk(retiredChunks_[i]);
            retiredChunks_.erase(retiredChunks_.begin() + static_cast<ptrdiff_t>(i));
        }
    }

    // (1) Throttle: wait until the GPU finished the frame that last used
    // this slot's command buffer and uniform buffer. This is the line that
    // enforces "at most kMaxFramesInFlight frames in flight" - and the
    // moral equivalent of the hidden stall WebGL2 applied somewhere inside
    // some gl call when you outran the GPU; here it has an address.
    vkWaitForFences(device_, 1, &inFlight_[currentFrame_], VK_TRUE, UINT64_MAX);

    // (2) Ask the presentation engine for an image to render into. This is
    // asynchronous: the call returns an INDEX immediately, but the image
    // itself is only guaranteed free once imageAvailable_ signals - on the
    // GPU timeline, not the CPU's. OUT_OF_DATE means the window changed
    // enough (resize, some minimizes) that the swapchain is unusable.
    uint32_t imageIndex = 0;
    const VkResult acquireResult =
        vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX,
                              imageAvailable_[currentFrame_], VK_NULL_HANDLE, &imageIndex);
    if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapchain();
        return false; // fence still signaled: deliberate, see (3)
    }
    if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR) {
        vkCheck(acquireResult, "vkAcquireNextImageKHR");
    }

    // (3) Reset the fence only now that we know we will submit work that
    // re-signals it. Resetting before the early-return above would deadlock
    // the next wait on this slot.
    vkResetFences(device_, 1, &inFlight_[currentFrame_]);

    updateUniformBuffer(currentFrame_, input);

    const bool capture = !pendingScreenshotPath_.empty();
    if (capture) prepareScreenshotBuffer();

    vkResetCommandBuffer(commandBuffers_[currentFrame_], 0);
    recordCommandBuffer(commandBuffers_[currentFrame_], imageIndex, input, capture);

    // (4) Submit: "wait for imageAvailable before the COLOR_ATTACHMENT
    // stage" - vertex work may start before the image is free, only the
    // actual pixel writes need to wait. Signal renderFinished[image] when
    // done, and signal this slot's fence so a future CPU frame can reuse it.
    const VkPipelineStageFlags waitStage =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &imageAvailable_[currentFrame_];
    submitInfo.pWaitDstStageMask = &waitStage;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers_[currentFrame_];
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &renderFinished_[imageIndex];
    vkCheck(vkQueueSubmit(queue_, 1, &submitInfo, inFlight_[currentFrame_]),
            "vkQueueSubmit");

    if (capture) {
        // Screenshot frames are deliberately synchronous: wait for this
        // frame's fence (fence-wait also makes the GPU's writes to coherent
        // memory visible to the host), then read the buffer out.
        vkWaitForFences(device_, 1, &inFlight_[currentFrame_], VK_TRUE, UINT64_MAX);
        saveScreenshot();
    }

    // (5) Present: the presentation engine waits for renderFinished[image]
    // before scanning the image out. The GPU may not even have started
    // rendering yet when this call returns - everything is chained through
    // semaphores.
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &renderFinished_[imageIndex];
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapchain_;
    presentInfo.pImageIndices = &imageIndex;
    const VkResult presentResult = vkQueuePresentKHR(queue_, &presentInfo);

    bool swapchainStale = framebufferResized_;
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR ||
        presentResult == VK_SUBOPTIMAL_KHR) {
        swapchainStale = true;
    } else if (presentResult != VK_SUCCESS) {
        vkCheck(presentResult, "vkQueuePresentKHR");
    }
    if (swapchainStale) {
        framebufferResized_ = false;
        recreateSwapchain();
    }

    currentFrame_ = (currentFrame_ + 1) % kMaxFramesInFlight;
    return true;
}

// ===========================================================================
// Screenshot capture.
// ===========================================================================

void Renderer::requestScreenshot(std::string path) {
    if (!supportsScreenshots_) {
        std::fprintf(stderr,
                     "screenshot unavailable: swapchain lacks TRANSFER_SRC usage\n");
        return;
    }
    pendingScreenshotPath_ = std::move(path);
}

void Renderer::prepareScreenshotBuffer() {
    const VkDeviceSize size =
        VkDeviceSize{4} * swapchainExtent_.width * swapchainExtent_.height;
    createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 screenshotBuffer_, screenshotMemory_);
}

void Renderer::saveScreenshot() {
    void* mapped = nullptr;
    vkCheck(vkMapMemory(device_, screenshotMemory_, 0, VK_WHOLE_SIZE, 0, &mapped),
            "vkMapMemory(screenshot)");
    const auto* pixels = static_cast<const uint8_t*>(mapped);

    // Swapchain formats are 4 bytes/pixel; only the channel order differs.
    const bool blueFirst = swapchainFormat_ == VK_FORMAT_B8G8R8A8_SRGB ||
                           swapchainFormat_ == VK_FORMAT_B8G8R8A8_UNORM;

    // Binary PPM (P6): dead-simple, dependency-free, readable by everything.
    // The sRGB-encoded bytes go out as-is - PPM viewers expect sRGB.
    std::ofstream file(pendingScreenshotPath_, std::ios::binary);
    if (file) {
        file << "P6\n"
             << swapchainExtent_.width << " " << swapchainExtent_.height << "\n255\n";
        const size_t pixelCount =
            size_t{swapchainExtent_.width} * swapchainExtent_.height;
        std::vector<uint8_t> row;
        row.reserve(pixelCount * 3);
        for (size_t i = 0; i < pixelCount; ++i) {
            const uint8_t* p = pixels + i * 4;
            row.push_back(blueFirst ? p[2] : p[0]);
            row.push_back(p[1]);
            row.push_back(blueFirst ? p[0] : p[2]);
        }
        file.write(reinterpret_cast<const char*>(row.data()),
                   static_cast<std::streamsize>(row.size()));
        std::printf("Saved screenshot: %s (%ux%u)\n", pendingScreenshotPath_.c_str(),
                    swapchainExtent_.width, swapchainExtent_.height);
    } else {
        std::fprintf(stderr, "failed to open %s for writing\n",
                     pendingScreenshotPath_.c_str());
    }

    vkUnmapMemory(device_, screenshotMemory_);
    // The fence wait in drawFrame proved the GPU is done with the buffer.
    vkDestroyBuffer(device_, screenshotBuffer_, nullptr);
    vkFreeMemory(device_, screenshotMemory_, nullptr);
    screenshotBuffer_ = VK_NULL_HANDLE;
    screenshotMemory_ = VK_NULL_HANDLE;
    pendingScreenshotPath_.clear();
}

// ===========================================================================
// Buffer helpers.
// ===========================================================================

// Vulkan separates "a buffer" (usage + size) from "the memory behind it".
// The memoryTypeBits mask says which memory types the buffer CAN live in;
// we intersect that with the properties we WANT (device-local, or
// host-visible) to pick a type index. WebGL2 exposed none of this - every
// gl.bufferData chose silently.
uint32_t Renderer::findMemoryType(uint32_t typeBits,
                                  VkMemoryPropertyFlags properties) const {
    VkPhysicalDeviceMemoryProperties memProps{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        const bool allowed = (typeBits & (1u << i)) != 0;
        const bool matches =
            (memProps.memoryTypes[i].propertyFlags & properties) == properties;
        if (allowed && matches) return i;
    }
    throw std::runtime_error("no suitable memory type found");
}

void Renderer::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                            VkMemoryPropertyFlags properties, VkBuffer& buffer,
                            VkDeviceMemory& memory) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCheck(vkCreateBuffer(device_, &bufferInfo, nullptr, &buffer), "vkCreateBuffer");

    VkMemoryRequirements memReq{};
    vkGetBufferMemoryRequirements(device_, buffer, &memReq);

    // Production note: one vkAllocateMemory per buffer is the teaching
    // version. Real engines allocate large blocks and sub-allocate (VMA is
    // the standard library for this) - drivers may cap total allocations at
    // 4096, and allocation itself is slow.
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, properties);
    vkCheck(vkAllocateMemory(device_, &allocInfo, nullptr, &memory), "vkAllocateMemory");
    vkCheck(vkBindBufferMemory(device_, buffer, memory, 0), "vkBindBufferMemory");
}

// The staging pattern: DEVICE_LOCAL memory is the fast memory next to the
// GPU cores, and on discrete cards the CPU (mostly) cannot write it. So:
// write into a temporary HOST_VISIBLE buffer, then record a GPU-side copy.
void Renderer::createDeviceLocalBuffer(const void* data, VkDeviceSize size,
                                       VkBufferUsageFlags usage, VkBuffer& buffer,
                                       VkDeviceMemory& memory) {
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 staging, stagingMemory);

    void* mapped = nullptr;
    vkCheck(vkMapMemory(device_, stagingMemory, 0, size, 0, &mapped), "vkMapMemory");
    std::memcpy(mapped, data, static_cast<size_t>(size));
    vkUnmapMemory(device_, stagingMemory);

    createBuffer(size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer, memory);

    VkCommandBuffer cmd = beginOneShotCommands();
    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(cmd, staging, buffer, 1, &copyRegion);
    endOneShotCommands(cmd);

    vkDestroyBuffer(device_, staging, nullptr);
    vkFreeMemory(device_, stagingMemory, nullptr);
}

// One-shot command buffer for setup-time transfers: record, submit, block on
// a fence until the GPU finishes. Blocking is fine at load time; streaming
// assets mid-game would instead keep the fence and poll it.
VkCommandBuffer Renderer::beginOneShotCommands() {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool_;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkCheck(vkAllocateCommandBuffers(device_, &allocInfo, &cmd),
            "vkAllocateCommandBuffers(one-shot)");

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    // The driver may optimize for "recorded once, submitted once, thrown
    // away" - exactly our usage.
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkCheck(vkBeginCommandBuffer(cmd, &beginInfo), "vkBeginCommandBuffer(one-shot)");
    return cmd;
}

void Renderer::endOneShotCommands(VkCommandBuffer cmd) {
    vkCheck(vkEndCommandBuffer(cmd), "vkEndCommandBuffer(one-shot)");

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    vkCheck(vkCreateFence(device_, &fenceInfo, nullptr, &fence), "vkCreateFence");

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    vkCheck(vkQueueSubmit(queue_, 1, &submitInfo, fence), "vkQueueSubmit(one-shot)");

    // A fence (not vkQueueWaitIdle) so we only wait for OUR submission -
    // waitIdle would also serialize against any other in-flight work.
    vkCheck(vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX),
            "vkWaitForFences(one-shot)");

    vkDestroyFence(device_, fence, nullptr);
    vkFreeCommandBuffers(device_, commandPool_, 1, &cmd);
}
