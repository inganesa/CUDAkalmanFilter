# Add source files here
EXECUTABLE	:= kalman
# Cuda source files (compiled with cudacc)
CUFILES		:= kalman.cu 
CUDEPS		:= matrix.cu matrix_kernel.cu 
CFLAGS          := -g

# C/C++ source files (compiled with gcc / c++)
CCFILES		:= invert.cpp
USECUBLAS       := 1
################################################################################
# Rules and targets

include ../../common/common.mk
