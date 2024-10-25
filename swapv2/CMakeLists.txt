cmake_minimum_required(VERSION 3.17)
project(cuda3 CUDA C CXX)

set(CMAKE_CXX_STANDARD 17)

# Set the path to the CUDA Toolkit
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.8)

# Find Python
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)

# Find Torch
set(CMAKE_PREFIX_PATH /home/lml/anaconda3/lib/python3.9/site-packages/torch/share/cmake)
find_package(Torch REQUIRED)

# Include directories
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(include)  # Add this line if there is an 'include' directory

# Source files
set(SRC_FILES
        swap.cpp
        main.cpp
        include/utils.h
        swap_kernal.cu
)

# Create the executable
add_executable(cuda3 ${SRC_FILES})

# Link libraries
target_link_libraries(cuda3 PRIVATE ${TORCH_LIBRARIES} ${PYTHON_LIBRARIES})

# Set CUDA properties for the executable
set_property(TARGET cuda3 PROPERTY CUDA_STANDARD 17)
set_property(TARGET cuda3 PROPERTY CUDA_SEPARABLE_COMPILATION ON)
