TARGET=main
OBJECTS=util.o mat_mul.o

CPPFLAGS=-std=c++14 -Wall -O3 -march=znver2
LDFLAGS=-lm -pthread

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)
