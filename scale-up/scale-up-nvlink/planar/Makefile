include ../../shared.mk

planar: planar_langford_multigpu.cu
	$(NVCC) $(NVCC_FLAGS) -std=c++11  $^ -o $@

clean: 
	rm -f planar
