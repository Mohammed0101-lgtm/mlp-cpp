cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(mlp)

find_package(Torch REQUIRED)

# Set CXX flags for compiling with Torch
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(mlp mlp.cpp)

# Link Torch libraries with the executable
target_link_libraries(mlp "${TORCH_LIBRARIES}")

# Set the C++ standard to C++17
set_property(TARGET mlp PROPERTY CXX_STANDARD 20)

# Specific workaround to remove unnecessary linker flags on macOS
if (APPLE)
  # Prevent adding unnecessary linker flags on macOS
  set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "")
endif()

# Windows-specific DLL copy handling
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET mlp
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:mlp>)
endif (MSVC)