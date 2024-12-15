cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(mlp)

# Find the Torch package
find_package(Torch REQUIRED)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the executable
add_executable(mlp mlp.cpp)

# Link Torch libraries
target_link_libraries(mlp "${TORCH_LIBRARIES}")

set_property(TARGET mlp PROPERTY CXX_STANDARD 17)

# Include Torch directories
target_include_directories(mlp PUBLIC ${TORCH_INCLUDE_DIRS})

# Add Torch compile options
target_compile_options(mlp PRIVATE "${TORCH_CXX_FLAGS}")
