CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++17

SRCS = memory_mountain.cpp analysis.cpp clock.cpp
OBJS = $(SRCS:.cpp=.o)
	EXEC = memory_mountain

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(EXEC) $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(EXEC)

