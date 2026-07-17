// xrsps-native: an OSRS world viewer in C++/Vulkan.
//
// This is a native port of a vertical slice of the XRSPS WebGL2 renderer
// (https://github.com/xrsps/xrsps-typescript). Throughout the code base the
// comments contrast the two APIs: WebGL2, where the driver hides resource
// and synchronization management, and Vulkan, where every one of those
// decisions is written out by hand.
//
// main.cpp owns the window, the input handling, and the frame loop; all
// Vulkan lives behind the Renderer class.

#include "camera.h"
#include "model.h"
#include "renderer.h"
#include "world.h"

#include <GLFW/glfw3.h>

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr int kInitialWidth = 1280;
constexpr int kInitialHeight = 768;

// Everything the GLFW callbacks need, reachable via the window user pointer
// (GLFW callbacks are C function pointers, so no lambdas with captures).
struct App {
    Renderer* renderer = nullptr;
    OrbitCamera camera;
    bool dragging = false;
    bool panning = false;
    double lastCursorX = 0.0;
    double lastCursorY = 0.0;
};

// Command-line options; all optional. The screenshot/self-test flags exist
// so correctness can be verified from scripts and CI, not just by eyeball.
struct Options {
    bool selfTest = false;
    std::string screenshotPath; // non-empty: capture after `frames`, then exit
    int frames = 8;
    bool hasYaw = false;
    float yaw = 0.0f;
    bool hasPitch = false;
    float pitch = 0.0f;
    bool hasDistance = false;
    float distance = 0.0f;
    std::string cachePath;   // empty: search ./cache then ../cache
    int mapX = 48;           // map-square coords; 48,54 = Edgeville
    int mapY = 54;
    int radius = 2;
};

void printUsage() {
    std::printf(
        "usage: xrsps-native [options]\n"
        "  --screenshot <path>  render, save a PPM screenshot, exit\n"
        "  --frames <n>         frames to render before the screenshot (default 8)\n"
        "  --yaw <deg>          initial camera yaw\n"
        "  --pitch <deg>        initial camera pitch\n"
        "  --dist <units>       initial camera distance\n"
        "  --self-test          scripted resize/minimize/screenshot run;\n"
        "                       exit code reflects the validation-message count\n"
        "  --cache <dir>        cache directory (default: ./cache or ../cache)\n"
        "  --map <mx,my>        starting map square (default 48,54 = Edgeville)\n"
        "  --radius <n>         streaming radius in map squares (default 2)\n");
}

bool parseOptions(int argc, char** argv, Options& options) {
    auto needValue = [&](int& i) -> const char* {
        if (i + 1 >= argc) {
            std::fprintf(stderr, "missing value for %s\n", argv[i]);
            return nullptr;
        }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (std::strcmp(arg, "--self-test") == 0) {
            options.selfTest = true;
        } else if (std::strcmp(arg, "--screenshot") == 0) {
            const char* value = needValue(i);
            if (value == nullptr) return false;
            options.screenshotPath = value;
        } else if (std::strcmp(arg, "--frames") == 0) {
            const char* value = needValue(i);
            if (value == nullptr) return false;
            options.frames = std::atoi(value);
        } else if (std::strcmp(arg, "--yaw") == 0) {
            const char* value = needValue(i);
            if (value == nullptr) return false;
            options.hasYaw = true;
            options.yaw = static_cast<float>(std::atof(value));
        } else if (std::strcmp(arg, "--pitch") == 0) {
            const char* value = needValue(i);
            if (value == nullptr) return false;
            options.hasPitch = true;
            options.pitch = static_cast<float>(std::atof(value));
        } else if (std::strcmp(arg, "--dist") == 0) {
            const char* value = needValue(i);
            if (value == nullptr) return false;
            options.hasDistance = true;
            options.distance = static_cast<float>(std::atof(value));
        } else if (std::strcmp(arg, "--cache") == 0) {
            const char* value = needValue(i);
            if (value == nullptr) return false;
            options.cachePath = value;
        } else if (std::strcmp(arg, "--map") == 0) {
            const char* value = needValue(i);
            if (value == nullptr) return false;
            if (std::sscanf(value, "%d,%d", &options.mapX, &options.mapY) != 2) {
                std::fprintf(stderr, "--map expects <mx,my>, e.g. --map 48,54\n");
                return false;
            }
        } else if (std::strcmp(arg, "--radius") == 0) {
            const char* value = needValue(i);
            if (value == nullptr) return false;
            options.radius = std::clamp(std::atoi(value), 0, 10);
        } else if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            printUsage();
            return false;
        } else {
            std::fprintf(stderr, "unknown option: %s\n", arg);
            printUsage();
            return false;
        }
    }
    return true;
}

// --- GLFW callbacks ---------------------------------------------------------

App* appFrom(GLFWwindow* window) {
    return static_cast<App*>(glfwGetWindowUserPointer(window));
}

void onFramebufferResized(GLFWwindow* window, int /*width*/, int /*height*/) {
    // Only flag it; the renderer recreates the swapchain at a safe point
    // inside drawFrame. (Callbacks can fire mid-frame on some platforms.)
    appFrom(window)->renderer->onFramebufferResized();
}

void onMouseButton(GLFWwindow* window, int button, int action, int /*mods*/) {
    App* app = appFrom(window);
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        app->dragging = (action == GLFW_PRESS);
        glfwGetCursorPos(window, &app->lastCursorX, &app->lastCursorY);
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        app->panning = (action == GLFW_PRESS);
        glfwGetCursorPos(window, &app->lastCursorX, &app->lastCursorY);
    }
}

void onCursorPos(GLFWwindow* window, double x, double y) {
    App* app = appFrom(window);
    const float dx = static_cast<float>(x - app->lastCursorX);
    const float dy = static_cast<float>(y - app->lastCursorY);
    if (app->dragging) {
        app->camera.rotate(dx, dy);
    } else if (app->panning) {
        app->camera.pan(dx, dy);
    }
    app->lastCursorX = x;
    app->lastCursorY = y;
}

void onScroll(GLFWwindow* window, double /*dx*/, double dy) {
    appFrom(window)->camera.zoom(static_cast<float>(dy));
}

void onKey(GLFWwindow* window, int key, int /*scancode*/, int action, int /*mods*/) {
    if (action != GLFW_PRESS) return;
    App* app = appFrom(window);
    switch (key) {
        case GLFW_KEY_F2:
            app->renderer->requestScreenshot("screenshot.ppm");
            break;
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, GLFW_TRUE);
            break;
        default:
            break;
    }
}

void onGlfwError(int code, const char* description) {
    std::fprintf(stderr, "[glfw] error %d: %s\n", code, description);
}

// Scripted actions for --self-test: exercises the paths that historically
// crash naive Vulkan apps (resize, minimize to 0x0, restore) plus the
// screenshot copy, then exits. Frame numbers are arbitrary; they just
// leave a few frames between actions.
void runSelfTestStep(GLFWwindow* window, App& app, uint64_t frame) {
    switch (frame) {
        case 20: glfwSetWindowSize(window, 1100, 700); break;
        case 45: glfwIconifyWindow(window); break;
        case 70: glfwRestoreWindow(window); break;
        case 120: app.renderer->requestScreenshot("selftest.ppm"); break;
        case 140: glfwSetWindowShouldClose(window, GLFW_TRUE); break;
        default: break;
    }
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parseOptions(argc, argv, options)) return 2;

    glfwSetErrorCallback(onGlfwError);
    if (!glfwInit()) {
        std::fprintf(stderr, "fatal: glfwInit failed\n");
        return 1;
    }

    // WebGL2 note: in the browser the window and the GL context arrive
    // together through canvas.getContext("webgl2"). GLFW splits those
    // concerns - we ask for a window with *no* client API attached, because
    // Vulkan hooks up to the window through its own object (VkSurfaceKHR)
    // instead of a context bound to the window's default framebuffer.
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    GLFWwindow* window =
        glfwCreateWindow(kInitialWidth, kInitialHeight, "xrsps-native", nullptr, nullptr);
    if (window == nullptr) {
        std::fprintf(stderr, "fatal: glfwCreateWindow failed\n");
        glfwTerminate();
        return 1;
    }

    int exitCode = 0;
    try {
        App app;

        // Real cache terrain, streamed: map squares load and unload around
        // the camera at runtime, so the base model stays empty and all
        // geometry draws through instance 0's identity transform.
        std::string cachePath = options.cachePath;
        if (cachePath.empty()) {
            for (const char* candidate : {"cache", "../cache"}) {
                if (std::filesystem::exists(std::filesystem::path(candidate) /
                                            "main_file_cache.dat2")) {
                    cachePath = candidate;
                    break;
                }
            }
            if (cachePath.empty()) {
                throw std::runtime_error(
                    "no cache found: put an OSRS cache dump in ./cache or pass --cache");
            }
        }

        WorldStreamer streamer;
        streamer.open(cachePath);
        std::printf("Streaming world around map square (%d, %d), radius %d\n",
                    options.mapX, options.mapY, options.radius);

        // World coordinates are absolute: viewer x = tile x, z = -tile y.
        app.camera.setTarget(glm::vec3(options.mapX * 64 + 32, 0.0f,
                                       -(options.mapY * 64 + 32)));
        app.camera.setAngles(options.hasYaw ? options.yaw : 30.0f,
                             options.hasPitch ? options.pitch : 40.0f);
        app.camera.setDistance(options.hasDistance ? options.distance : 90.0f);

        Model model; // stays empty - all geometry arrives as streamed chunks
        std::vector<InstanceData> instances;
        instances.push_back({glm::mat4(1.0f)});
        Renderer renderer(window, model, instances, streamer.textures());
        app.renderer = &renderer;

        // Streamed squares: key -> chunk handle (0 = square known absent).
        std::unordered_map<int64_t, uint64_t> loadedSquares;
        auto squareKey = [](int mx, int my) {
            return (static_cast<int64_t>(mx) << 32) |
                   static_cast<uint32_t>(static_cast<int32_t>(my));
        };

        glfwSetWindowUserPointer(window, &app);
        glfwSetFramebufferSizeCallback(window, onFramebufferResized);
        glfwSetMouseButtonCallback(window, onMouseButton);
        glfwSetCursorPosCallback(window, onCursorPos);
        glfwSetScrollCallback(window, onScroll);
        glfwSetKeyCallback(window, onKey);

        std::printf(
            "controls: drag LMB = orbit | drag RMB / WASD = pan | scroll = zoom | "
            "F2 = screenshot.ppm | ESC = quit\n");

        uint64_t frameNumber = 0;
        double lastTitleTime = glfwGetTime();
        double lastFrameTime = lastTitleTime;
        double smoothedMs = 0.0;

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            // WASD flies the orbit target across the ground plane.
            const float panStep =
                static_cast<float>(std::max(smoothedMs, 8.0)) * 0.6f;
            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) app.camera.pan(0.0f, panStep);
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) app.camera.pan(0.0f, -panStep);
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) app.camera.pan(panStep, 0.0f);
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) app.camera.pan(-panStep, 0.0f);

            // Stream map squares around the camera target: drop the ones
            // far outside the radius, then build the nearest missing one
            // (one per frame keeps the frame time steady - a production
            // streamer would build on worker threads instead).
            const glm::vec3 target = app.camera.target();
            const int centerX = static_cast<int>(std::floor(target.x / 64.0f));
            const int centerY = static_cast<int>(std::floor(-target.z / 64.0f));
            const int radius = options.radius;

            for (auto it = loadedSquares.begin(); it != loadedSquares.end();) {
                const int mx = static_cast<int>(it->first >> 32);
                const int my = static_cast<int32_t>(it->first & 0xffffffff);
                if (std::abs(mx - centerX) > radius + 1 ||
                    std::abs(my - centerY) > radius + 1) {
                    renderer.removeChunkGeometry(it->second);
                    it = loadedSquares.erase(it);
                } else {
                    ++it;
                }
            }

            int bestX = 0;
            int bestY = 0;
            int bestDistance = INT_MAX;
            for (int mx = centerX - radius; mx <= centerX + radius; ++mx) {
                for (int my = centerY - radius; my <= centerY + radius; ++my) {
                    if (loadedSquares.count(squareKey(mx, my)) != 0) continue;
                    const int d = (mx - centerX) * (mx - centerX) +
                                  (my - centerY) * (my - centerY);
                    if (d < bestDistance) {
                        bestDistance = d;
                        bestX = mx;
                        bestY = my;
                    }
                }
            }
            if (bestDistance != INT_MAX) {
                Model chunk;
                uint64_t handle = 0;
                if (streamer.buildSquare(bestX, bestY, chunk)) {
                    handle = renderer.addChunkGeometry(chunk);
                }
                loadedSquares.emplace(squareKey(bestX, bestY), handle);
            }

            if (options.selfTest) runSelfTestStep(window, app, frameNumber);
            if (!options.screenshotPath.empty()) {
                if (frameNumber == static_cast<uint64_t>(options.frames)) {
                    renderer.requestScreenshot(options.screenshotPath);
                } else if (frameNumber > static_cast<uint64_t>(options.frames) + 2) {
                    glfwSetWindowShouldClose(window, GLFW_TRUE);
                }
            }

            // Aspect ratio comes from the swapchain, not the window: during
            // a resize they briefly disagree, and the swapchain is what we
            // actually render into.
            const VkExtent2D extent = renderer.swapchainExtent();
            const float aspect =
                extent.height == 0
                    ? 1.0f
                    : static_cast<float>(extent.width) / static_cast<float>(extent.height);

            FrameInput input{};
            input.view = app.camera.viewMatrix();
            input.proj = app.camera.projectionMatrix(aspect);
            input.instanceCount = 1; // instance 0: the identity transform
            input.firstInstance = 0;
            // World colors already contain the game's baked hillshade;
            // make the shader's lighting a no-op.
            input.ambient = 1.0f;
            input.diffuse = 0.0f;

            renderer.drawFrame(input);

            // CPU frame-time readout in the title bar (exponential moving
            // average). CPU time only - GPU time would need timestamp
            // queries; see STUDY.md.
            const double now = glfwGetTime();
            const double frameMs = (now - lastFrameTime) * 1000.0;
            lastFrameTime = now;
            smoothedMs = smoothedMs == 0.0 ? frameMs : smoothedMs * 0.95 + frameMs * 0.05;
            if (now - lastTitleTime > 0.5) {
                lastTitleTime = now;
                // Sized for the worst case: deviceName alone may be 255
                // characters (VK_MAX_PHYSICAL_DEVICE_NAME_SIZE).
                char title[384];
                std::snprintf(title, sizeof(title),
                              "xrsps-native | %s | %.2f ms (%.0f fps)",
                              renderer.deviceName(), smoothedMs,
                              smoothedMs > 0.0 ? 1000.0 / smoothedMs : 0.0);
                glfwSetWindowTitle(window, title);
            }

            ++frameNumber;
        }

        renderer.waitIdle();
        if (renderer.validationMessageCount() > 0) {
            std::fprintf(stderr, "%llu validation message(s) - see log above\n",
                         static_cast<unsigned long long>(renderer.validationMessageCount()));
            exitCode = 1;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        exitCode = 1;
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return exitCode;
}
