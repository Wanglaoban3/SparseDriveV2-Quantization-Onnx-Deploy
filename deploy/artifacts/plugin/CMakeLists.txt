cmake_minimum_required(VERSION 3.18)
project(dfa_plugin LANGUAGES CXX CUDA)

# Board-side build:
#   cmake -B build -DTENSORRT_ROOT=/path/to/TensorRT -DCMAKE_CUDA_ARCHITECTURES=<sm of your board>
#   cmake --build build -j
# The resulting libdfa_plugin.so is loaded with ctypes.CDLL() before parsing the ONNX.

set(TENSORRT_ROOT "" CACHE PATH "TensorRT root (with include/ and lib/)")

add_library(dfa_plugin SHARED dfa_plugin.cu)

target_include_directories(dfa_plugin PRIVATE
    ${TENSORRT_ROOT}/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
target_link_directories(dfa_plugin PRIVATE
    ${TENSORRT_ROOT}/lib
    ${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}
)
target_link_libraries(dfa_plugin PRIVATE nvinfer cudart)
target_compile_options(dfa_plugin PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
