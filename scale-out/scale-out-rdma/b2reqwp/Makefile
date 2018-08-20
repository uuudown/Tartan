include ../../shared.mk

b2r:  b2r.cpp b2r.h eqwp.cu eqwp.hpp  b2r_halo_sync.cu
	$(NVCC) $(NVCC_FLAGS) $(LINK_FLAG) b2r.cpp eqwp.cu b2r_halo_sync.cu  -o $@

clean:
	rm -r b2r






