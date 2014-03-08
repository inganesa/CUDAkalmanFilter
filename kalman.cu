

//Includes
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cutil_inline.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "matrix_kernel.cu"
#include "matrix.cu"
#include <cublas.h>

extern void invert(float * A, int lda, int n);

//Function declaration
int kalmanFilter(int ns, int no);
void cleanup(void);
void ParseArguments(int, char**);

// create and start timer as unsigned integer
unsigned int timer_mem = 0;
unsigned int timer_total = 0;
unsigned int timer_GPU = 0;
unsigned int timer_CPU = 0;

//Variable declarations
int ns = 10, no = 5; 
int dev = 0;

float *X;//Estimate
float *h_X;//Estimate
float *P;//Uncertainity Covariance t
float *F;//State transition Matrix
float *Z;//Measurement
float *S;//Intermediate value
float *s;//Intermediate value
float *si;//Intermediate value
float *K;//Kalman gain
float *H;//Measurement function
float *E;//Measurement noise
float *Ft;//F transpose
float *Ht;//H transpose
float *Si;//Inverse of S
float *Y; //error
float *I;//Identity Matrix
float *Hint;//for intermediate calculations
float *Sint;//for intermediate calculations
float *Kint;//for intermediate calculations
float *Xint;// for intermediate calculations
float *Pint;// for intermediate calculations
float *Pint2;// for intermediate calculations
float *Ztemp;//to  store temporarily
int meslen = 1;//number of iterations


//Device Variable declarations
float *d_X;//Estimate
float *d_P;//Uncertainity Covariance
float *d_F;//State transition Matrix
float *d_Z;//Measurement
float *d_S;//Intermediate value
float *d_s;//Intermediate value
float *d_K;//Kalman gain
float *d_H;//Measurement function
float *d_E;//Measurement noise
float *d_Ft;//F transpose
float *d_Ht;//H transpose
float *d_Si;//Inverse of S
float *d_Y; //error
float *d_I;//Identity Matrix
float *d_Hint;//for intermediate calculations
float *d_Sint;//for intermediate calculations
float *d_Kint;//for intermediate calculations
float *d_Xint;// for intermediate calculations
float *d_Pint;// for intermediate calculations
float *d_Pint2;// for intermediate calculations
float *d_Ztemp;//to  store temporarily

//Host Code

int main(int argc, char** argv)
{
int success;
ParseArguments(argc, argv);

printf("\nKalman Filter : ns %d no: %d \n", ns, no);

//Initialize the timer to zero cycles.
cutilCheckError(cutCreateTimer(&timer_CPU));
cutilCheckError(cutCreateTimer(&timer_mem));
cutilCheckError(cutCreateTimer(&timer_total));
cutilCheckError(cutCreateTimer(&timer_GPU));
	
//Memory Allocation
memAlloc(&X,ns,1);
memAlloc(&h_X,ns,1);
memAlloc(&P,ns,ns);
memAlloc(&F,ns,ns);
memAlloc(&Z,no,1);
memAlloc(&S,no,no);
memAlloc(&s,no,no);
memAlloc(&si,no,no);
memAlloc(&K,ns,no);
memAlloc(&H,no,ns);
memAlloc(&E,no,no);
memAlloc(&Ft,ns,ns);
memAlloc(&Ht,ns,no);
memAlloc(&Si,no,no);
memAlloc(&Y,no,1);
memAlloc(&I,ns,ns);
memAlloc(&Hint,no,ns);
memAlloc(&Sint,no,no);
memAlloc(&Kint,ns,no);
memAlloc(&Xint,ns,1);
memAlloc(&Pint,ns,ns);
memAlloc(&Pint2,ns,ns);
Ztemp = Z;
printf("\nHost allocation is completed...\n");

Initialize(X,P,F,Z,H,E,I,Ht,Ft,s,ns,no);

printf("\nInitialization of the host variables are completed...\n");

//Allocate vectors  in device memory
cudaMalloc(&d_X, ns*1);
cudaMalloc(&d_P, ns*ns);
cudaMalloc(&d_F, ns*ns);
cudaMalloc(&d_Z, no*1);
cudaMalloc(&d_S, no*no);
cudaMalloc(&d_s, no*no);
cudaMalloc(&d_K, ns*no);
cudaMalloc(&d_H, no*ns);
cudaMalloc(&d_E, no*no);
cudaMalloc(&d_Ft, ns*ns);
cudaMalloc(&d_Ht, ns*no);
cudaMalloc(&d_Si, no*no);
cudaMalloc(&d_Y, no*1);
cudaMalloc(&d_I, ns*ns);
cudaMalloc(&d_Hint, no*ns);
cudaMalloc(&d_Sint, no*no);
cudaMalloc(&d_Kint, ns*no);
cudaMalloc(&d_Xint, ns*1);
cudaMalloc(&d_Pint, ns*ns);
cudaMalloc(&d_Pint2, ns*ns);
float *A;
    int lda = ((no+15)&~15|16);
	
	cudaError_t ret = cudaMallocHost( (void**)&A, no*lda*sizeof(float) );
    if( ret != cudaSuccess ) {
      printf("Failed to allocate %d memory", no*lda*sizeof(float));
      return 1;
    }
	
printf("\nAllocation of the Device memory completed...\n");

for(int i = 0; i<no ; i++)
	{
	for(int j = i; j < no; j++)
		{
		A[i*lda + j] = s[i * no + j];
		A[j*lda+i] = A[i*lda+j];
		}
	}

	
// Start the timer
cutilCheckError(cutStartTimer(timer_mem));
	
// Copy Input vectors from host memory to device memory
cudaMemcpy(d_X, X, ns*1, cudaMemcpyHostToDevice);
cudaMemcpy(d_P, P, ns*ns, cudaMemcpyHostToDevice);
cudaMemcpy(d_F, F, ns*ns, cudaMemcpyHostToDevice);
cudaMemcpy(d_Z, Z, no*1, cudaMemcpyHostToDevice);
cudaMemcpy(d_S, S, no*no, cudaMemcpyHostToDevice);
cudaMemcpy(d_H, H, no*ns, cudaMemcpyHostToDevice);
cudaMemcpy(d_E, E, no*no, cudaMemcpyHostToDevice);
cudaMemcpy(d_I, I, ns*ns, cudaMemcpyHostToDevice);
	
// stop timer
cutilCheckError(cutStopTimer(timer_mem));

// Print the timer
printf("\nCPU to GPU Transfer Time: %f (ms)\n", cutGetTimerValue(timer_mem));
	
//Start the CPU timer
cutilCheckError(cutStartTimer(timer_CPU));

success = kalmanFilter(ns, no);
if (success)
printf( "\nKalman Filter CPU Execution Successful!!! \n ");
 
//Stop the CPU timer  
cutilCheckError(cutStopTimer(timer_CPU));
printf("\nCPU Execution Time: %f (ms) \n", cutGetTimerValue(timer_CPU));

// Set the kernel arguments
int threadsPerBlock = 256;
int Nos = no*ns;
int Ns =  ns;
int No = no;
int Ns2 = ns*ns;
int No2 = no*no;
int blocksPerGridNos = (Nos + threadsPerBlock - 1) / threadsPerBlock;
int blocksPerGridNs = (Ns + threadsPerBlock - 1) / threadsPerBlock;
int blocksPerGridNo = (No + threadsPerBlock - 1) / threadsPerBlock;
int blocksPerGridNs2 = (Ns2 + threadsPerBlock - 1) / threadsPerBlock;
int blocksPerGridNo2 = (No2 + threadsPerBlock - 1) / threadsPerBlock;

	
//Invoke kernel 
	// Start the timer
	cutilCheckError(cutStartTimer(timer_total));    
	
	// Start GPU timer
	cutilCheckError(cutStartTimer(timer_GPU));

//Inverse using CUBLAS
	
    if( cudaSetDevice( dev ) != cudaSuccess )
    {
      printf( "Failed to set device %d\n", dev );
      return 1;
    }
    
	if( cublasInit( ) != CUBLAS_STATUS_SUCCESS )
    {
      printf( "failed to initialize the cublas library\n" );
      return 1;
    }
	printf("Cublas initialized...\n");

	invert(A, lda, no);

	for(int i = 0; i<no ; i++)
		{	
		for(int j = i; j < no; j++)
			{
			si[i * no + j] = A[i * lda + j];
			si[j * no + i] = si[i * no + j];
			}			
		}
	
	cudaMemcpy(d_Si, si, no*no, cudaMemcpyHostToDevice); 
	
//step 1  to calculate Y = Z - HX
MatMult<<<blocksPerGridNos, threadsPerBlock>>>(d_Y, d_H, d_X, no, ns);
MatSub<<<blocksPerGridNo, threadsPerBlock>>>(d_Y, d_Z, d_Y, no, 1);
	
//step 2 to calculate  S = HPHt + E
	
MatMult<<<blocksPerGridNos, threadsPerBlock>>>(d_Hint, d_H, d_P, no, ns);
MatMult<<<blocksPerGridNos, threadsPerBlock>>>(d_Sint, d_Hint, d_Ht, no, ns);
MatAdd<<<blocksPerGridNo2, threadsPerBlock>>>(d_S, d_Sint, d_E, no, no);
	
//step 3 to calcualte K = PHtSi 
	
MatMult<<<blocksPerGridNos, threadsPerBlock>>>(d_Kint, d_P, d_Ht, no, ns);

MatMult<<<blocksPerGridNos, threadsPerBlock>>>(d_K, d_Kint, d_Si, ns, no);
	
//step4 to calculate  X = X+ KY

MatMult<<<blocksPerGridNos, threadsPerBlock>>>(d_Xint, d_K, d_Y, ns, no);
MatAdd<<<blocksPerGridNs, threadsPerBlock>>>(d_X, d_X, d_Xint, ns, 1);
	
//step5 to calculate [I - KH]P
	
MatMult<<<blocksPerGridNos, threadsPerBlock>>>(d_Pint, d_K, d_H, ns, no);
MatSub<<<blocksPerGridNs2, threadsPerBlock>>>(d_Pint, d_I, d_Pint, ns, ns);
MatMult<<<blocksPerGridNs2, threadsPerBlock>>>(d_Pint2, d_Pint, d_P, ns, ns);
MatCopy<<<blocksPerGridNs2, threadsPerBlock>>>(d_P, d_Pint2, ns, ns);
	
//Prediction Phase
// X = FX
// P = FPFt 
	
//step 1 to calculate X = FX
MatMult<<<blocksPerGridNs2, threadsPerBlock>>>(d_Xint, d_F, d_X, ns, ns);
MatCopy<<<blocksPerGridNs, threadsPerBlock>>>(d_X, d_Xint, ns, 1);
	
//step2 to calculate P = FPFt 
	
MatMult<<<blocksPerGridNs2, threadsPerBlock>>>(d_Pint, d_F, d_P, ns, ns);
MatMult<<<blocksPerGridNs2, threadsPerBlock>>>(d_P, d_Pint, d_Ft, ns, ns);

// Host wait for the kernel to finish
cudaThreadSynchronize(); 

// stop GPU timer
cutilCheckError(cutStopTimer(timer_GPU));
	
// Reset the timer for memory
cutilCheckError(cutCreateTimer(&timer_mem));

// Start the timer
cutilCheckError(cutStartTimer(timer_mem));
		
// Copy result from device memory to host memory
// X contains the result in host memory
cudaMemcpy(h_X, d_X, ns*1, cudaMemcpyDeviceToHost);
    	
// stop and destroy timer
cutilCheckError(cutStopTimer(timer_mem));
cutilCheckError(cutStopTimer(timer_total));

cudaFreeHost( A );
cublasShutdown();

printf( "\nKalman Filter GPU Execution Successful!!! \n ");

// Print the timer
printf("\nGPU to CPU Transfer Time: %f (ms) \n", cutGetTimerValue(timer_mem));
printf("\nOverall Execution Time (Memory + GPU): %f (ms) \n", cutGetTimerValue(timer_total));
printf("\nGPU Execution Time: %f (ms) \n", cutGetTimerValue(timer_GPU));

cleanup();

} 

// Kalman Filter Equations.

int kalmanFilter(int ns, int no)
{
int i;
int success;
for (i = 0 ; i < meslen ; i++)
{
	
//Update
//S = HPHt + E
//K = PHtSi 
//Y = Z-HX
//X = X + KY
//P = [I - KH]P
	
	//step1  Y = Z-HX
	mult(Y,H,X,no,ns,ns,1);
	
	sub(Y,Z,Y,no,1);
	
	//step2 S = HPHt + E
	mult(Hint,H,P,no,ns,ns,ns);
	
	mult(Sint,Hint,Ht,no,ns,ns,no);
					
	add(S,Sint,E,no,no);
	
		
	//step3 K = PHtSi
	mult(Kint,P,Ht,ns,ns,ns,no);
	
	success = inverse(no,s,Si);
	
	if( success == 0)
	{
	 printf ("\nExecution failed \n ");
	 return 0;
	}
	else
	{
	printf("\n CPU Inversion successfull\n");
	}
	
	mult(K,Kint,Si,ns,no,no,no);
	
	//printf("\nUpdate phase Step 3 successfull\n");
	
	//step4  X = X + KY
	mult(Xint,K,Y,ns,no,no,1);

	add(X,X,Xint,ns,1);
	
	//printf("\nUpdate phase Step 4 successfull\n");
	
	//step5  P = [I - KH]P
	mult(Pint,K,H,ns,no,no,ns);

	sub(Pint,I,Pint,ns,ns);
	
	mult(Pint2,Pint,P,ns,ns,ns,ns);

	matcopy(P,Pint2,ns,ns);
	
	//printf("\nUpdate phase Step 5 successfull\n");

	//Prediction Phase
    // X = FX
	// P = FPFt 
	
	//step 1 X = FX
	mult(Xint,F,X,ns,ns,ns,1);
	
	matcopy(X,Xint,ns,1);
	
	//printf("\nPredict phase Step 1 successfull\n");
	
	//step 2  P = FPFt 
	mult(Pint, F, P,ns,ns,ns,ns);
	
	mult(P,Pint,Ft,ns,ns,ns,ns);
			
	//printf("\nPredict phase Step 2 successfull\n");
	
	Z = &Z[no];
   }
   return 1;
}


//clean up resources
void cleanup(void)
{
	
// Free host memory
if(X)
	free(X);
if(P)
	free(P);
if(F)
	free(F);
if(Ztemp)
	free(Ztemp);
if(S)
	free(S);
if(s)
	free(s);
if(K)
	free(K);
if(H)
	free(H);
if(E)
	free(E);
if(Ft)
	free(Ft);
if(Ht)
	free(Ht);
if(Si)
	free(Si);
if(Y)
	free(Y);
if(I)
	free(I);
if(Hint)
	free(Hint);
if(Sint)
	free(Sint);
if(Kint)
	free(Kint);
if(Xint)
	free(Xint);
if(Pint)
	free(Pint);
if(Pint2)
	free(Pint2);
//printf("\n Host Cleanup Successful\n");

/*
// Free device memory
 
	if(d_X)
		free(d_X);
	if(d_P)
		free(d_P);
		
	if(d_F)
		free(d_F);
	if(d_Ztemp)
		free(d_Ztemp);
	if(d_S)
		free(d_S);
	if(d_s)
		free(d_s);
	if(d_K)
		free(d_K);
	if(d_H)
		free(d_H);
	if(d_E)
		free(d_E);
	if(d_Ft)
		free(d_Ft);
	if(d_Ht)
		free(d_Ht);
	if(d_Si)
		free(d_Si);
	if(d_Y)
		free(d_Y);
	if(d_I)
		free(d_I);
	if(d_Hint)
		free(d_Hint);
	if(d_Sint)
		free(d_Sint);
	if(d_Kint)
		free(d_Kint);
	if(d_Xint)
		free(d_Xint);
	if(d_Pint)
		free(d_Pint);
	if(d_Pint2)
		free(d_Pint2);
		
	printf("\nDevice Cleanup Successful\n");  */
	
    // Destroy (Free) timer   
    cutilCheckError(cutDeleteTimer(timer_mem));
    cutilCheckError(cutDeleteTimer(timer_total));
    cutilCheckError(cutDeleteTimer(timer_GPU));
	cutilCheckError(cutDeleteTimer(timer_CPU));
      
    cutilSafeCall( cudaThreadExit() );
    
    exit(0); 
}
 
// Parse program arguments
void ParseArguments(int argc, char** argv)
{
    for (int i = 0; i < argc; ++i) 
		{
        if (strcmp(argv[i], "--ns") == 0 || strcmp(argv[i], "-ns") == 0) 
			{
			ns = atoi(argv[i+1]);
			i = i + 1;
			}
		}
	
    for (int i = 0; i < argc; ++i) 
		{
        if (strcmp(argv[i], "--no") == 0 || strcmp(argv[i], "-no") == 0) 
			{
			no = atoi(argv[i+1]);
			i = i + 1;
			}
		}
	for (int i = 0; i < argc; ++i) 
		{
        if (strcmp(argv[i], "--dev") == 0 || strcmp(argv[i], "-dev") == 0) 
			{
			dev = atoi(argv[i+1]);
			i = i + 1;
			}
		}
}

 
