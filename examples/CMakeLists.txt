# Example executables for sam3.cpp
# SDL2 + ImGui are optional — build only if SDL2 is found.

find_package(SDL2 QUIET)

if(SDL2_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third-party/CMakeLists.txt")
    add_subdirectory(third-party)

    add_executable(sam3_image main_image.cpp)
    target_link_libraries(sam3_image PRIVATE sam3 imgui-sdl2 SDL2::SDL2)

    add_executable(sam3_video main_video.cpp)
    target_link_libraries(sam3_video PRIVATE sam3 imgui-sdl2 SDL2::SDL2)
else()
    message(STATUS "SDL2 not found or third-party/ missing — skipping GUI examples")
endif()
