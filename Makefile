CXX      = g++
CXXFLAGS = -O2 -Wall -Wextra -std=c++14 -Isrc -Ithird_party
LDFLAGS  = -lmpfr -lgmp -ldl

SHARED_SRC = src/reference.cpp
SHARED_OBJ = src/reference.o

GEMM_TARGET  = gemm_tester
TRSM_TARGET  = trsm_tester
MUMPS_TARGET = mumps_tester

GEMM_OBJ  = src/gemm_tester.o
TRSM_OBJ  = src/trsm_tester.o
MUMPS_OBJ = src/mumps_tester.o

.PHONY: all clean

all: $(GEMM_TARGET) $(TRSM_TARGET) $(MUMPS_TARGET)

$(GEMM_TARGET): $(GEMM_OBJ) $(SHARED_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(TRSM_TARGET): $(TRSM_OBJ) $(SHARED_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(MUMPS_TARGET): $(MUMPS_OBJ) $(SHARED_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

src/reference.o: src/reference.cpp src/reference.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

src/gemm_tester.o: src/gemm_tester.cpp src/reference.h src/tester_utils.h third_party/CLI11.hpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

src/trsm_tester.o: src/trsm_tester.cpp src/reference.h src/tester_utils.h third_party/CLI11.hpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

src/mumps_tester.o: src/mumps_tester.cpp src/reference.h src/tester_utils.h third_party/CLI11.hpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(SHARED_OBJ) $(GEMM_OBJ) $(TRSM_OBJ) $(MUMPS_OBJ) $(GEMM_TARGET) $(TRSM_TARGET) $(MUMPS_TARGET)
