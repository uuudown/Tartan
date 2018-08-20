include ../../shared.mk

kmeans: centroids.cu kmeans.cu labels.cu test.cu timer.cu 
	$(NVCC) $(NVCC_FLAGS) $(LINK_FLAG) -lcublas $^ -o $@

clean: 
	rm -f kmeans
