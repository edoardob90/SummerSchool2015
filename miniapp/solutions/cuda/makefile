CXX=g++
CUDA=nvcc
CPPFLAGS=-I${CUDA_INCLUDE}
CXXFLAGS=-O3 -fopenmp -L${CUDA_LIBRARY}
CUDAFLAGS=-O3 -std=c++11 -arch=sm_35
LIBS=-lcublas -lcudart
INFO="NOT USING"

SOURCES = stats.cu  data.cu  operators.cu  linalg.cu    main.cu
HEADERS = stats.h   data.h   operators.h   linalg.h  cuda_helpers.h
OBJ     = stats.o   data.o   operators.o   linalg.o

ifneq ($(SHARED),)
	INFO="USING"
	CPPFLAGS+=-D__USE_STENCIL_SHARED
	CUDAFLAGS+=-D__USE_STENCIL_SHARED
endif

.SUFFIXES: .cpp

all: main unit_tests
	@echo "$(INFO) shared stencil kernel" >&2

stats.o: stats.cu $(HEADERS)
	$(CUDA) $(CUDAFLAGS) -c stats.cu

data.o: data.cu $(HEADERS)
	$(CUDA) $(CUDAFLAGS) -c data.cu

operators.o: operators.cu $(HEADERS)
	$(CUDA) $(CUDAFLAGS) -c operators.cu

linalg.o: linalg.cu $(HEADERS)
	$(CUDA) $(CUDAFLAGS) -c linalg.cu

main.o: main.cu $(HEADERS)
	$(CUDA) $(CUDAFLAGS) -c main.cu

unit_tests.o: unit_tests.cu $(HEADERS)
	$(CUDA) $(CUDAFLAGS) -c unit_tests.cu

main: $(OBJ) main.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OBJ) main.o -o main $(LIBS)

unit_tests: data.o linalg.o unit_tests.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) unit_tests.o $(OBJ) -o unit_tests $(LIBS)
	./unit_tests

clean:
	rm -f main
	rm -f unit_tests
	rm -f *.o
	rm -f *.i
	rm -f output.*
