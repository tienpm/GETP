CUDA_INSTALL_PATH = /usr/local/cuda/include  # Adjust if CUDA is installed elsewhere
OPENCL_INSTALL_PATH = /opt/AMDAPP    # Adjust if OpenCL is installed elsewhere

NVCC = $(CUDA_INSTALL_PATH)/bin/nvcc
CXX = g++

TARGET=opencl_info

# Additional flags (adjust as needed)
CFLAGS = -Wall -g
NVCCFLAGS = -arch=sm_61  # Replace with your GPU architecture
LDLIBS = -lOpenCL

SRC = opencl_info.cpp
OBJ = $(SRC:.cpp=.o)

%.o: %.cpp
    $(CXX) $(CFLAGS) -c $< -o $@

%.o: %.cu
    $(NVCC) $(NVCCFLAGS) -c $< -o $@

all: $(TARGET)

# ${TARGET}: $(OBJ)
#     $(NVCC) $(NVCCFLAGS) -o $@ $^ -lOpenCL

${TARGET}: $(OBJ)
    $(CXX) -I$(CUDA_INSTALL_PATH) $(LDLIBS) -o $@ $^
clean:
	rm -f $(TARGET) $(OBJ) *.o
