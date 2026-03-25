cmake_minimum_required(VERSION 3.14)
project(sam3.cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ggml options — enable Metal on Apple
option(SAM3_METAL "Enable Metal backend" ON)
if(APPLE AND SAM3_METAL)
    set(GGML_METAL        ON CACHE BOOL "" FORCE)
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
endif()

add_subdirectory(ggml)

# sam3 static library
add_library(sam3 STATIC sam3.cpp sam3.h)
target_include_directories(sam3 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/stb
)
target_link_libraries(sam3 PUBLIC ggml)
target_compile_features(sam3 PUBLIC cxx_std_14)

# Examples (only when top-level)
option(SAM3_BUILD_EXAMPLES "Build example executables" ON)
if(SAM3_BUILD_EXAMPLES AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    add_subdirectory(examples)
endif()

# Tests
option(SAM3_BUILD_TESTS "Build test executables" OFF)
if(SAM3_BUILD_TESTS)
    add_subdirectory(tests)
endif()
