cmake_minimum_required(VERSION 3.25)
project(mcrt_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(vulkan_headers
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
    GIT_TAG v1.4.363
    GIT_SHALLOW TRUE)
FetchContent_Declare(volk
    GIT_REPOSITORY https://github.com/zeux/volk.git
    GIT_TAG vulkan-sdk-1.4.363.0
    GIT_SHALLOW TRUE)
FetchContent_Declare(vma
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG v3.4.0
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(vulkan_headers)

set(VOLK_PULL_IN_VULKAN OFF CACHE BOOL "" FORCE)
set(VOLK_HEADERS_ONLY ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(volk vma)

# ---- Shaders: Slang -> SPIR-V -> embedded C++ arrays ----------------------------------
set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../shaders)
find_program(SLANGC slangc HINTS ${CMAKE_CURRENT_SOURCE_DIR}/../tools/slang/bin REQUIRED)
set(GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GEN_DIR}/shaders)

set(SHADERS pathtrace denoise sky)
set(SHADER_HEADERS)
foreach(name IN LISTS SHADERS)
    set(spv ${GEN_DIR}/shaders/${name}.spv)
    set(hdr ${GEN_DIR}/shaders/${name}.spv.h)
    file(GLOB_RECURSE SHADER_DEPS ${SHADER_DIR}/*.slang)
    add_custom_command(
        OUTPUT ${hdr}
        COMMAND ${SLANGC} ${SHADER_DIR}/${name}.slang -target spirv -profile spirv_1_5
                -fvk-use-entrypoint-name -matrix-layout-column-major -O2 -o ${spv}
        COMMAND ${CMAKE_COMMAND} -DINPUT=${spv} -DOUTPUT=${hdr} -DSYMBOL=kSpirv_${name}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spirv.cmake
        DEPENDS ${SHADER_DEPS} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spirv.cmake
        COMMENT "Compiling shader ${name}.slang"
        VERBATIM)
    list(APPEND SHADER_HEADERS ${hdr})
endforeach()
add_custom_target(mcrt_shaders DEPENDS ${SHADER_HEADERS})

# ---- Native library ---------------------------------------------------------------------
add_library(mcrt_native SHARED
    src/api.cpp
    src/bevel.cpp
    src/denoiser.cpp
    src/material_store.cpp
    src/renderer.cpp
    src/section_manager.cpp
    src/shapes.cpp
    src/terrain_mesher.cpp
    src/vk_context.cpp
    src/vma_impl.cpp)
add_dependencies(mcrt_native mcrt_shaders)
target_include_directories(mcrt_native PRIVATE include src ${GEN_DIR})
target_link_libraries(mcrt_native PRIVATE Vulkan::Headers volk::volk_headers GPUOpen::VulkanMemoryAllocator)
target_compile_definitions(mcrt_native PRIVATE
    VK_NO_PROTOTYPES
    VMA_STATIC_VULKAN_FUNCTIONS=0
    VMA_DYNAMIC_VULKAN_FUNCTIONS=1
    NOMINMAX WIN32_LEAN_AND_MEAN)
if(MSVC)
    target_compile_options(mcrt_native PRIVATE /W4 /permissive- /utf-8)
    set_source_files_properties(src/vma_impl.cpp PROPERTIES COMPILE_OPTIONS /W0)
    # Static CRT: no redistributable needed on players' machines.
    set_property(TARGET mcrt_native PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()
