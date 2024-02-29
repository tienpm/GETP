LDLIB_INCLUDE = /usr/local/cuda/include 
LDLIB_LIB = /usr/local/cuda/lib 

CXX = mpic++
NVCC = $(CUDA_ROOT)/bin/nvcc
CFLAGS = -Wall -g -I$(LDLIB_INCLUDE) #--show-me
LDFLAGS = -lOpenCL
# NVCCFLAGS = -arch=sm_61 -I$(LDLUB_INCLUDE)

TARGET = main
SRC = main.cpp kernel.cl
OBJ = main.o kernel.o 

all: $(TARGET) 

$TARGET: $(OBJ)
	$(CXX) $(LDLIB_INCLUDE) $(CFLAGS) $(LDFLAGS) -o $@ # -L$(LDLUB_LIB) -lld_rt

%.o: %.cpp
	$(CXX) $(LDLIB_INCLUDE) $(CFLAGS) $(LDFLAGS) -c $< -o $@

# %.o: %.cu
#     $(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJ)
