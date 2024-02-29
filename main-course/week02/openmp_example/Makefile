# Compiler choice
CXX = mpic++

# Compiler flags (adjust as needed)
CXXFLAGS = -std=c++11 -Wall -O3 -march=znver2 -fopt-info-vec # --showme
LDFLAGS = -fopenmp -fcf-protection=none -fno-stack-protector # -lm -lmpi_cxx -lmpi

# Target executable name
TARGET = main

# Source files
SOURCES = main.cpp

# Object files (automatically generated from source files)
OBJECTS = $(SOURCES:.cpp=.o)

# Default target (builds the executable)
all: $(TARGET)

# Rule to build the executable
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(OBJECTS) -o $@

# Rule to build object files from source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -c $< -o $@

# Clean target to remove intermediate files
clean:
	rm -f $(OBJECTS) $(TARGET)
