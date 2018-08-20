include ../../shared.mk

# double precision (ON/OFF)
DOUBLE_PRECISION = ON
# MPI for parallel (ON/OFF)
DO_MPI = ON
# Maximum number of atoms that can be stored in a link cell
MAXATOMS = 64
# Assertions within device kernels as well as sanity-checks scattered within the C code. (ON/OFF)
DEBUG = OFF

# nvcc-specific options
NVCC_FLAGS += -DMAXATOMS=$(MAXATOMS) -DDO_MPI -I$(MPI_DIR)/include -L$(MPI_DIR)/lib

# add precision flag for CUDA targets
ifeq ($(DOUBLE_PRECISION), ON)
NVCC_FLAGS += -DCOMD_DOUBLE
else
NVCC_FLAGS += -DCOMD_SINGLE
endif

ifeq ($(DEBUG), ON)
CFLAGS += -DDEBUG -O0 -g
NVCC_FLAGS += -O0 -DDEBUG
else
# specify optimization flags
CFLAGS += -DNDEBUG ${OPTFLAGS}
NVCC_FLAGS += -DNDEBUG
endif

MPICC_FLAGS += -std=c99 -Wno-unused-result -DMAXATOMS=$(MAXATOMS) -DDO_MPI -I$(CUDA_DIR)/include

# Check for double precision
ifeq ($(DOUBLE_PRECISION), ON)
MPICC_FLAGS += -DCOMD_DOUBLE
else
MPICC_FLAGS += -DCOMD_SINGLE
endif

# c files to be compiled with mpicc
SOURCES=$(wildcard *.c)
OBJECTS=$(SOURCES:.c=.o)

# cuda files to be compiled with nvcc
CU_SOURCES=$(wildcard *.cu)
CU_OBJECTS=$(CU_SOURCES:.cu=.cu.o)

DEFAULT: CoMD

%.cu.o: %.cu
	${NVCC} -c $< ${NVCC_FLAGS} -o $@

%.o: %.c
	${MPICC} ${MPICC_FLAGS} -c $< -o $@

CoMD: ${BIN_DIR} CoMD_info.h ${OBJECTS} ${CU_OBJECTS}
	${MPICC} ${MPICC_FLAGS} -o $@ ${OBJECTS} ${CU_OBJECTS} -lcudart $(LINK_FLAG)

CoMD_info.h: Makefile
	./generate_info_header ${CoMD_VARIANT} "$(MPICC)" "$(MPICC_FLAGS)" "$(LINK_FLAGS)"

clean:
	rm -f *.o CoMD_info.h CoMD


