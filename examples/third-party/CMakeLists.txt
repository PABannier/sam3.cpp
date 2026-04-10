# ImGui library with SDL2 + OpenGL3 backend.
#
# ImGui is not vendored. Prefer a local checkout at examples/third-party/imgui
# (useful for offline builds); otherwise fetch it at configure time via
# FetchContent, pinned to a known-good tag.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/imgui/imgui.cpp")
    set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/imgui)
    message(STATUS "Using local ImGui checkout at ${IMGUI_DIR}")
else()
    include(FetchContent)
    FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG        v1.91.8
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(imgui)
    set(IMGUI_DIR ${imgui_SOURCE_DIR})
    message(STATUS "Fetched ImGui v1.91.8 into ${IMGUI_DIR}")
endif()

add_library(imgui-sdl2 STATIC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_sdl2.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

target_include_directories(imgui-sdl2 PUBLIC
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)

target_link_libraries(imgui-sdl2 PUBLIC SDL2::SDL2)

# OpenGL
if(APPLE)
    target_link_libraries(imgui-sdl2 PUBLIC "-framework OpenGL")
else()
    find_package(OpenGL REQUIRED)
    target_link_libraries(imgui-sdl2 PUBLIC OpenGL::GL)
endif()

# On non-Apple, leave IMGUI_IMPL_OPENGL_LOADER_* undefined so imgui uses its
# bundled imgl3w loader (required for GL 3+ on MSVC where opengl32.lib only
# exports GL 1.1 symbols).
if(APPLE)
    target_compile_definitions(imgui-sdl2 PUBLIC GL_SILENCE_DEPRECATION)
endif()
