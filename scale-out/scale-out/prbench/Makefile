include ../../shared.mk

prbench: main.cpp phsort.c cuda_kernels.cu adtp.c utils.c tmp_pool.c
	$(NVCC) $(NVCC_FLAGS) $(LINK_FLAG) -I./cub/ $^ -o $@
clean:
	rm -rf *.o prbench
