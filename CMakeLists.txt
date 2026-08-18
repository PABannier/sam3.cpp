cmake_minimum_required(VERSION 3.14)
project(sam3.cpp VERSION 1.0.0 LANGUAGES C CXX)

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

# Check if CUDA should be enabled
option(SAM3_CUDA "Enable CUDA backend" OFF)
# ggml options — enable CUDA
if(SAM3_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
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

# --- Install ---
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(SAM3_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Location of sam3 header files")
set(SAM3_LIB_INSTALL_DIR     ${CMAKE_INSTALL_LIBDIR}     CACHE PATH "Location of sam3 library files")
set(SAM3_BIN_INSTALL_DIR     ${CMAKE_INSTALL_BINDIR}     CACHE PATH "Location of sam3 binary files")

install(TARGETS sam3 ARCHIVE DESTINATION ${SAM3_LIB_INSTALL_DIR})
install(FILES sam3.h DESTINATION ${SAM3_INCLUDE_INSTALL_DIR})
install(FILES stb/stb_image.h stb/stb_image_write.h DESTINATION ${SAM3_INCLUDE_INSTALL_DIR}/stb)

configure_package_config_file(
    cmake/sam3-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/sam3-config.cmake
    INSTALL_DESTINATION ${SAM3_LIB_INSTALL_DIR}/cmake/sam3
    PATH_VARS SAM3_INCLUDE_INSTALL_DIR SAM3_LIB_INSTALL_DIR SAM3_BIN_INSTALL_DIR)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/sam3-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/sam3-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/sam3-version.cmake
    DESTINATION ${SAM3_LIB_INSTALL_DIR}/cmake/sam3)

# Package archive (make package / cpack)
set(CPACK_PACKAGE_NAME "sam3")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_VENDOR "sam3.cpp")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "C++ inference for Meta SAM 3 on ggml")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "sam3.cpp developers")
set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "C++ inference for Meta SAM 3 on ggml")
if(WIN32)
    set(CPACK_GENERATOR "ZIP")
else()
    set(CPACK_GENERATOR "TGZ;DEB")
endif()
if(NOT CPACK_PACKAGE_FILE_NAME)
    set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME}")
endif()
include(CPack)

