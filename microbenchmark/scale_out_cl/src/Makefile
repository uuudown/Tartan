#
# Copyright (c) 2015-2017, NVIDIA CORPORATION. All rights reserved.
#
# See LICENCE.txt for license information
#

NCCL_HOME = /ccs/home/angli/tartan/Collective/nccl_2.0/
MPI = 1
MPI_HOME = /autofs/nccs-svm1_sw/summitdev/.swci/1-compute/opt/spack/20171006/linux-rhel7-ppc64le/xl-20170914-beta/spectrum-mpi-10.1.0.4-20170915-nmlgpsufnxxal2wv64hh7zfisabr56ry/

CUDA_HOME = /sw/summitdev/cuda/8.0.61-1/
PREFIX = /usr/local
VERBOSE = 0
DEBUG = 0

CUDA_LIB = $(CUDA_HOME)/lib64
CUDA_INC = $(CUDA_HOME)/include
NVCC = $(CUDA_HOME)/bin/nvcc

# Better define NVCC_GENCODE in your environment to the minimal set
# of archs to reduce compile time.
NVCC_GENCODE = -gencode=arch=compute_60,code=sm_60

NVCUFLAGS  := -ccbin $(CXX) $(NVCC_GENCODE) -std=c++11

LDFLAGS    := -L${CUDA_LIB} -lcudart -lrt
NVLDFLAGS  := -L${CUDA_LIB} -lcudart -lrt

ifeq ($(DEBUG), 0)
NVCUFLAGS += -O3
CXXFLAGS  += -O3
else
NVCUFLAGS += -O0 -G -g
CXXFLAGS  += -O0 -g -ggdb3
endif

ifeq ($(VERBOSE), 0)
.SILENT:
endif

.PHONY: build clean

BUILDDIR = ../build

ifneq ($(NCCL_HOME), "")
NVCUFLAGS += -I$(NCCL_HOME)/include/
NVLDFLAGS   += -L$(NCCL_HOME)/lib
endif

ifeq ($(MPI), 1)
NVCUFLAGS += -DMPI_SUPPORT -I$(MPI_HOME)/include
NVLDFLAGS += -L$(MPI_HOME)/lib -lmpi_ibm
endif
LIBRARIES += curand nccl_static nvToolsExt
NVLDFLAGS   += $(LIBRARIES:%=-l%)

DST_DIR := $(BUILDDIR)
SRC_FILES := $(wildcard *.cu)
OBJ_FILES := $(SRC_FILES:%.cu=${DST_DIR}/%.o)
BIN_FILES_LIST := all_reduce all_gather broadcast reduce_scatter reduce
BIN_FILES := $(BIN_FILES_LIST:%=${DST_DIR}/%_perf)

build: ${BIN_FILES}

clean:
	rm -rf ${DST_DIR}/*_perf

${DST_DIR}/%.o: %.cu
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) -c $<

${DST_DIR}/%_perf:${DST_DIR}/%.o ${DST_DIR}/common.o
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ ${NVLDFLAGS}

