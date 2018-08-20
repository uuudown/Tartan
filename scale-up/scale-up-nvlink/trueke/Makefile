##############################################################
# Makefile                                                   #
#                                                            #
# Author      : Cristobal Navarro <crinavar@dcc.uchile.cl>   #
# Version     : 1.0                                          #
# Date        : September 2015                               #
# Description : Compiles the program                         #
##############################################################



include ../../shared.mk


# hierarchy, names
OBJ		:= ./obj
SRC		:= ./src
DEP		:= ./dep
EXEC	:= trueke
CXXFLAGS   := -O3 -g -fopenmp

# paths
CUDA_ROOT = /usr/local/cuda
CUDA_INSTALL_PATH = $(CUDA_ROOT)
CUDA_SDK_PATH = $(CUDA_ROOT)/samples

# compiler, include and lib parameters
INCD = -I"$(CUDA_SDK_PATH)/common/inc" -I"$(CUDA_INSTALL_PATH)/include" -I"./" -I"/usr/include/nvidia/gdk/"
LIBS = -lcuda -L"$(CUDA_INSTALL_PATH)/lib64" -lcudart -L"$(CUDA_SDK_PATH)/common/lib"  -lpthread -lnvidia-ml -lgomp -L$(CUDA_DRIVER_DIR) -lnccl

# compiler flags
SPECIAL_FLAGS = --compiler-options -fno-strict-aliasing,-O3,-march=native,-msse4,-mfpmath=sse,-funroll-loops,-finline-functions -Xcompiler -fopenmp -Xcompiler -std=c++98
NVCCFLAGS	:= -m64 -lineinfo  ${SPECIAL_FLAGS} -D_FORCE_INLINES
DEBUG_NVCCFLAGS := --ptxas-options=-v -G -g-ccbin /usr/bin/g++ -Xptxas -dlcm=cg

# source files
CPP_SOURCES	:= $(wildcard $(SRC)/*.cpp)
CPP_HEADERS	:= $(wildcard $(SRC)/*.h)
CU_SOURCES	:= $(wildcard $(SRC)/*.cu)
CU_HEADERS	:= $(wildcard $(SRC)/*.cuh)

# object files
CPP_OBJS	:= $(patsubst $(SRC)/%.cpp, $(OBJ)/%.o, $(CPP_SOURCES))
CU_OBJS		:= $(patsubst $(SRC)/%.cu, $(OBJ)/%.cuo, $(CU_SOURCES))

# dependencies
CPP_DEPS	:= $(patsubst $(OBJ)/%.o, $(DEP)/%.d,$(CPP_OBJS))
CU_DEPS		:= $(patsubst $(OBJ)/%.cuo, $(DEP)/%.cud,$(CU_OBJS))

all: $(EXEC)


$(OBJ)/%.cuo : $(SRC)/%.cu
	$(NVCC) $(NVCCFLAGS) $(NVCC_FLAGS) -c $(INCD) -o $@ $<
	$(NVCC) $(NVCCFLAGS) $(NVCC_FLAGS)  $(INCD) -M $(SRC)/$*.cu -MT $(OBJ)/$*.cuo > $(DEP)/$*.cud

$(OBJ)/%.o: $(SRC)/%.cpp
	$(CXX) -c $(CXXFLAGS) $(INCD) -o $@ $<
	$(CXX) -M $(SRC)/$*.cpp -MT $(OBJ)/$*.o > $(DEP)/$*.d

# rules 
$(EXEC): $(CPP_OBJS) $(CU_OBJS)
	$(CXX) -o $(EXEC) $(CU_OBJS) $(CPP_OBJS) $(LDFLAGS) $(INCD) $(LIBS)

clean:
	rm -f $(EXEC) $(OBJ)/*.o $(OBJ)/*.cuo $(DEP)/*.cud $(DEP)/*.d

# include dependency files
-include $(CU_DEPS)
-include $(CPP_DEPS)
