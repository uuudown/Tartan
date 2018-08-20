include ../../shared.mk

lulesh: allocator.cu vector.h lulesh.cu util.h vector.h texture_objAPI.h allocator.h
	$(NVCC) $(NVCC_FLAGS) -I. $(LINK_FLAG)  allocator.cu lulesh.cu -o $@

#-I/usr/lib/openmpi/include -L/usr/lib/openmpi/lib


clean: 
	rm -rf lulesh




