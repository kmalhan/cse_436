cmake_minimum_required(VERSION 2.6.4)
project(OpenMP)

# Steps to compile codes using CMake
# 1. Change directory to build
# 2. cmake ..
# 3. make
# 4. Run executable you want to run

# Bring the headers into project
include_directories(include)

# Set compiler flag
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")

# This can be used to add all at once (not used)
# file(GLOB SOURCES "src/*.c")
# add_executable(testRun ${SOURCES})

# Manually add sources for now
add_executable(hello src/hello.c)
add_executable(axpy src/axpy.c)
