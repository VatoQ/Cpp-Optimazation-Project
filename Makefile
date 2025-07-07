CXX := g++
CXXFLAGS := -Wall -g -O3 -ffast-math -funroll-loops -march=native -fopenmp -std=c++17
OPTFLAGS := -fopt-info-vec-all=opt_report.txt
LDFLAGS := -fopenmp

TARGET := main

SRCS = batch_multiplication.cpp

OBJS := $(SRCS:.cpp=.o)

all: $(TARGET)


$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $(TARGET) $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)
