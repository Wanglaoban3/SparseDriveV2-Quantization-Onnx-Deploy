# Linux build of the DFA TensorRT plugin.
#
#   TENSORRT_ROOT : directory containing the NvInfer*.h headers
#   CUDA_ARCH     : SM arch of the target GPU, e.g. 60/70/75/80/86/89/90
#   LINK_NVINFER  : empty by default -- with TRT 8.6 headers the plugin links
#                   WITHOUT libnvinfer (pure-virtual interface, verified);
#                   set to "-L<libdir> -lnvinfer" if your TRT build needs it.
#
# Example:
#   make TENSORRT_ROOT=/path/to/TensorRT-8.6-headers CUDA_ARCH=86
#
# The resulting libdfa_plugin.so is loaded with ctypes.CDLL() before parsing
# the ONNX; the plugin registers itself statically (REGISTER_TENSORRT_PLUGIN).

CXXCUDA       ?= nvcc
TENSORRT_ROOT ?= /usr/local/TensorRT
CUDA_ARCH     ?= 75
LINK_NVINFER  ?=

all: libdfa_plugin.so

libdfa_plugin.so: dfa_plugin.cu
	$(CXXCUDA) -O3 -std=c++17 -shared -Xcompiler -fPIC \
		--generate-code=arch=compute_$(CUDA_ARCH),code=[sm_$(CUDA_ARCH),compute_$(CUDA_ARCH)] \
		-I$(TENSORRT_ROOT)/include $(LINK_NVINFER) \
		-o $@ $<

clean:
	rm -f libdfa_plugin.so

.PHONY: all clean
