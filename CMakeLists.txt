# xrsps-native: OSRS world viewer in C++/Vulkan.
#
# The build intentionally avoids renderer frameworks and helper libraries
# (no vk-bootstrap, no VMA): the point of the project is that every Vulkan
# call is written out and explainable. The only dependencies are the window
# system glue (GLFW), header-only math (GLM), and the Vulkan SDK itself.
cmake_minimum_required(VERSION 3.21)

project(xrsps-native
    VERSION 0.1.0
    DESCRIPTION "OSRS world viewer in C++/Vulkan"
    # C is only needed when the zlib/bzip2 fallbacks below build from source.
    LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Single-config generators (Makefiles, Ninja) default to Debug so a fresh
# checkout runs with validation layers enabled; configure with
# -DCMAKE_BUILD_TYPE=Release for benchmarking. Multi-config generators
# (Visual Studio) pick the configuration at build time instead.
get_property(IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT IS_MULTI_CONFIG AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# --- Dependencies ------------------------------------------------------------

# The Vulkan SDK (or distro packages) provides the headers, the loader
# library, and the glslc shader compiler used further down.
find_package(Vulkan REQUIRED)

# All other dependencies come from system packages when available; otherwise
# they are downloaded as source archives (plain HTTPS - no git needed) and
# built as part of this project. Net effect: on Windows the only manual
# installs are the Vulkan SDK and CMake itself.
include(FetchContent)

find_package(glfw3 3.3 QUIET)
if(NOT glfw3_FOUND)
    message(STATUS "glfw3 not found on the system - fetching 3.4 sources")
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(glfw
        URL https://github.com/glfw/glfw/releases/download/3.4/glfw-3.4.zip)
    FetchContent_MakeAvailable(glfw)
endif()

find_package(glm QUIET)
if(NOT glm_FOUND)
    message(STATUS "glm not found on the system - fetching 1.0.1 sources")
    FetchContent_Declare(glm
        URL https://github.com/g-truc/glm/archive/refs/tags/1.0.1.zip)
    FetchContent_MakeAvailable(glm)
endif()
if(NOT TARGET glm::glm)
    # Some older system packages export the target as plain "glm".
    add_library(glm::glm ALIAS glm)
endif()

# zlib and bzip2 decompress the JS5 cache containers. Both
# are a handful of C files with no configure step, so when no system package
# exists the pristine upstream sources are downloaded and built as small
# static libraries right here - deliberately not vendored into the repo.
function(xrsps_fetch_third_party name url header)
    set(extractRoot ${CMAKE_BINARY_DIR}/_deps/${name}-src)
    file(GLOB found ${extractRoot}/*/${header})
    if(NOT found)
        message(STATUS "${name} not found on the system - fetching sources")
        set(archive ${CMAKE_BINARY_DIR}/_deps/${name}.archive)
        file(DOWNLOAD ${url} ${archive} STATUS downloadStatus)
        list(GET downloadStatus 0 downloadCode)
        if(NOT downloadCode EQUAL 0)
            message(FATAL_ERROR "Failed to download ${name}: ${downloadStatus}")
        endif()
        file(ARCHIVE_EXTRACT INPUT ${archive} DESTINATION ${extractRoot})
        file(GLOB found ${extractRoot}/*/${header})
        if(NOT found)
            message(FATAL_ERROR "${name} archive did not contain ${header}")
        endif()
    endif()
    list(GET found 0 headerPath)
    get_filename_component(sourceDir ${headerPath} DIRECTORY)
    set(${name}_SOURCE_DIR ${sourceDir} PARENT_SCOPE)
endfunction()

find_package(ZLIB QUIET)
if(NOT ZLIB_FOUND)
    xrsps_fetch_third_party(zlib
        https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz
        zlib.h)
    add_library(xrsps_zlib STATIC
        ${zlib_SOURCE_DIR}/adler32.c ${zlib_SOURCE_DIR}/compress.c
        ${zlib_SOURCE_DIR}/crc32.c ${zlib_SOURCE_DIR}/deflate.c
        ${zlib_SOURCE_DIR}/gzclose.c ${zlib_SOURCE_DIR}/gzlib.c
        ${zlib_SOURCE_DIR}/gzread.c ${zlib_SOURCE_DIR}/gzwrite.c
        ${zlib_SOURCE_DIR}/infback.c ${zlib_SOURCE_DIR}/inffast.c
        ${zlib_SOURCE_DIR}/inflate.c ${zlib_SOURCE_DIR}/inftrees.c
        ${zlib_SOURCE_DIR}/trees.c ${zlib_SOURCE_DIR}/uncompr.c
        ${zlib_SOURCE_DIR}/zutil.c)
    target_include_directories(xrsps_zlib PUBLIC ${zlib_SOURCE_DIR})
    # Third-party code is built as-is; its warnings are not our quality bar.
    target_compile_options(xrsps_zlib PRIVATE $<IF:$<CXX_COMPILER_ID:MSVC>,/w,-w>)
    add_library(ZLIB::ZLIB ALIAS xrsps_zlib)
endif()

find_package(BZip2 QUIET)
if(NOT BZIP2_FOUND)
    xrsps_fetch_third_party(bzip2
        https://sourceware.org/pub/bzip2/bzip2-1.0.8.tar.gz
        bzlib.h)
    add_library(xrsps_bzip2 STATIC
        ${bzip2_SOURCE_DIR}/blocksort.c ${bzip2_SOURCE_DIR}/bzlib.c
        ${bzip2_SOURCE_DIR}/compress.c ${bzip2_SOURCE_DIR}/crctable.c
        ${bzip2_SOURCE_DIR}/decompress.c ${bzip2_SOURCE_DIR}/huffman.c
        ${bzip2_SOURCE_DIR}/randtable.c)
    target_include_directories(xrsps_bzip2 PUBLIC ${bzip2_SOURCE_DIR})
    target_compile_options(xrsps_bzip2 PRIVATE $<IF:$<CXX_COMPILER_ID:MSVC>,/w,-w>)
    add_library(BZip2::BZip2 ALIAS xrsps_bzip2)
endif()

# --- Application -------------------------------------------------------------

add_executable(xrsps-native
    src/main.cpp
    src/renderer.cpp
    src/camera.cpp
    src/cache.cpp
    src/world.cpp)

target_link_libraries(xrsps-native PRIVATE
    Vulkan::Vulkan glfw glm::glm ZLIB::ZLIB BZip2::BZip2)

target_compile_definitions(xrsps-native PRIVATE
    # glfw3.h includes the OpenGL headers by default (its GL heritage
    # showing). This is a Vulkan-only application, so opt out; sources
    # include <vulkan/vulkan.h> themselves before glfw3.h where needed.
    GLFW_INCLUDE_NONE
    # GLM configuration - defined globally because every translation unit
    # must agree on them (they change inline function definitions):
    #   RADIANS: reject the degree-based legacy API entirely.
    #   DEPTH_ZERO_TO_ONE: Vulkan clip-space depth is [0, 1], not OpenGL's
    #   [-1, 1]; without this glm::perspective wastes half the depth range.
    GLM_FORCE_RADIANS
    GLM_FORCE_DEPTH_ZERO_TO_ONE)

# --- Shaders: GLSL -> SPIR-V at build time ------------------------------------
# WebGL2 shipped GLSL source and compiled it in the driver at runtime
# (gl.compileShader, per user, per run, with per-driver error quirks).
# Vulkan consumes pre-compiled SPIR-V, so shader compilation becomes an
# ordinary build step with build-time errors. glslc is shaderc's compiler
# and ships with the Vulkan SDK; glslangValidator would be the equivalent
# alternative (different CLI, same output).
if(NOT Vulkan_GLSLC_EXECUTABLE)
    find_program(Vulkan_GLSLC_EXECUTABLE glslc)
endif()
if(NOT Vulkan_GLSLC_EXECUTABLE)
    message(FATAL_ERROR "glslc not found - install the Vulkan SDK (or the glslc package)")
endif()

set(SHADER_SOURCES
    shaders/model.vert
    shaders/model.frag)

set(SPIRV_DIR ${CMAKE_CURRENT_BINARY_DIR}/spv)
set(SPIRV_FILES "")
foreach(SHADER IN LISTS SHADER_SOURCES)
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    set(SPV ${SPIRV_DIR}/${SHADER_NAME}.spv)
    add_custom_command(
        OUTPUT ${SPV}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SPIRV_DIR}
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} --target-env=vulkan1.3
                ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER} -o ${SPV}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER}
        COMMENT "glslc: ${SHADER_NAME} -> ${SHADER_NAME}.spv"
        VERBATIM)
    list(APPEND SPIRV_FILES ${SPV})
endforeach()

add_custom_target(shaders DEPENDS ${SPIRV_FILES})
add_dependencies(xrsps-native shaders)

# Copy the compiled shaders next to the executable, wherever the generator
# put it (build/ for Makefiles and Ninja, build/Debug/ etc. for Visual
# Studio). The app loads them relative to its own path at runtime.
add_custom_command(TARGET xrsps-native POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${SPIRV_DIR}
            $<TARGET_FILE_DIR:xrsps-native>/shaders
    VERBATIM)

# Warnings are part of the quality bar: the project builds clean at these
# levels on GCC, Clang, and MSVC.
if(MSVC)
    target_compile_options(xrsps-native PRIVATE /W4)
else()
    target_compile_options(xrsps-native PRIVATE -Wall -Wextra)
endif()
