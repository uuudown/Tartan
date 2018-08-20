include ../../shared.mk

MPICXX_FLAGS =  -m64 -O3 -Wall -fopenmp -funroll-loops $(NVCC_INCLUDE)
NVCC_FLAGS += -m64 -Xcompiler -fopenmp -Xcompiler -fno-strict-aliasing -Xcompiler -funroll-loops

all: diffusion

Kernels.o: Kernels.cu
	$(NVCC)  $(NVCC_FLAGS) -o $@ -c $<

Util.o: Util.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ -c $<

Tools.o: Tools.c
	$(MPICXX) $(MPICXX_FLAGS) -o $@ -c $<

Main.o: main.c
	$(MPICXX) $(MPICXX_FLAGS) -o $@ -c $<

diffusion: Main.o Tools.o Util.o Kernels.o
	$(MPICXX) -o $@ $+ $(MPICXX_FLAGS) $(LINK_FLAG) -lcudart
			
clean:
	rm -rf *.vtk *.o diffusion *.txt *.bin *.nvprof
