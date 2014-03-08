This package contains the implemenatation of Kalman Filter using a GPU.

For usage, see the file in kalman.cu

The matrix functions  used the CPU implementations are inlcuded in matrix.cu

The kernel codes for the GPU implementation are inlcuded in the matrix_kernel.co

This folder contains a function for inverting a symmetric positive definit matrix
using a GPU. For usage see the file inverse.cpp

The executable file is : kalman

Sample run:

$make
$kalman -ns 1000 -no 250

Kalman Filter : ns 1000 no: 250 

Host allocation is completed...

Initialization of the host variables are completed...

Allocation of the Device memory completed...

CPU to GPU Transfer Time: 2.447000 (ms)

CPU Inversion successfull

Kalman Filter CPU Execution Successful!!! 
 
CPU Execution Time: 17017.128906 (ms) 

Cublas initialized...
inversion started     

Kalman Filter GPU Execution Successful!!! 
 
GPU to CPU Transfer Time: 0.020000 (ms) 

Overall Execution Time (Memory + GPU): 14.652 (ms) 

GPU Execution Time: 12.195000 (ms)

