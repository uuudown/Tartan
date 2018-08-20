include ../../shared.mk

cusimann: minimizeSchwefel.cu
	$(NVCC) $(NVCC_FLAGS) -Xcompiler -fopenmp $(LINK_FLAG) $^  -lcurand -lnlopt_cxx -Lnlopt/lib -o $@

clean: 
	rm -f cusimann

