include ../../shared.mk

matvec: 
	$(NVCC) $(NVCC_FLAGS) $(LINK_FLAG)  ${MPI}  -lpthread matvec.cu -o $@

clean:
	rm -f matvec


