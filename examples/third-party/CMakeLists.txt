# ImGui library with SDL2 + OpenGL3 backend
set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/imgui)

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

if(APPLE)
    target_compile_definitions(imgui-sdl2 PUBLIC GL_SILENCE_DEPRECATION)
else()
    target_compile_definitions(imgui-sdl2 PUBLIC IMGUI_IMPL_OPENGL_LOADER_CUSTOM)
endif()
