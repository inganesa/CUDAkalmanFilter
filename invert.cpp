 
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas.h>

#define VERIFY 1

// size of panels can impact performance, try some values
#define PANEL_SIZE 32

inline void __cudaSafeCall( int err, const char *file, const int line )
{
  do {
    if( err != 0) {
      fprintf(stderr, "cudaSafeCall() Runtime API error in file <%s>, line %i : %s.\n",
	      file, line, cudaGetErrorString((cudaError_t) err) );
      exit(-1);
    }
  } while (0);
}

#define SAFECALL(err)           __cudaSafeCall      (err, __FILE__, __LINE__)

/* implementatin of Cholesky decomposition on the CPU */
void choleskyDecompositionCPU(int n, float * A, int lda) {
  //#define A(x,y) a[y*lda+x]
  float * L = new float[n*n];
  memset(L,0,sizeof(float)*n*n);

  for (int j = 0; j < n; j++) {
    float d = 0.0;
    for (int k = 0; k < j; k++) {
      float s = 0.0;
      for (int i = 0; i < k; i++) {
	s += L[k*n+i] * L[j*n+i];
      }
      L[j*n+k] = s = (A[j+k*lda] - s) / L[k*n+k];
      L[k*n+j] = L[j*n+k];
      d = d + s * s;
    }
    d = A[j*(lda+1)] - d;
    L[j*n+j] = (d > 0 ? sqrt(d) : 0);
  }
  for(int j = 0;j < n; j++) {
    for (int k = 0; k < n; k++) {
      A[j+k*lda] = L[j*n+k];
    }
  }
  delete [] L;
  return;
  //#undef A
} // choleskyDecompositionCPU

void choleskyDecompositionCPU2(int n, float * a, int lda) {
#define A(x,y) a[y*lda+x]
		float * L = new float[n*n];
		memset(L,0,sizeof(float)*n*n);

		for (int j = 0; j < n; j++) {
			float d = 0.0;
			for (int k = 0; k < j; k++) {
				float s = 0.0;
				for (int i = 0; i < k; i++) {
					s += L[k*n+i] * L[j*n+i];
				}
				L[j*n+k] = s = (A(j,k) - s) / L[k*n+k];
				L[k*n+j] = L[j*n+k];
				d = d + s * s;
			}
			d = A(j,j) - d;
			L[j*n+j] = (d > 0 ? sqrt(d) : 0);
		}
		for(int j = 0;j < n; j++) {
			for (int k = 0; k < n; k++) {
				A(j,k) = L[j*n+k];
			}
		}
		delete [] L;
		return;
#undef A
} // choleskyDecompositionCPU

void solveCPU ( const int m, const int n,
        const float *A, const int lda, float *B,
             const int ldb) {
  int i, j, k;
  //int n1, n2;
    for (i = 0; i < n; i++) {
        float Aii = A[lda * i + i];
        for (j = 0; j < m; j++) {
          B[ldb * i + j] /= Aii;
	 }

      for (k = i + 1; k < n; k++) {
        const float Aik = A[i * lda + k];
        for (j = 0; j < m; j++) {
          B[ldb * k + j] -= Aik * B[ldb * i + j];
        }
      }
    }
} // solveCPU
             	
/* implementation of Cholesky decomposition on the GPU 
 * following Vasily Volkov and James Demmel.  */
void choleskyDecompositionGPU( int n, float *cpu_A, float * A, int lda)
{	
    //const int BLOCK_SIZE = 128;
    SAFECALL(cudaMemcpy(A, cpu_A, n*lda*sizeof(float),cudaMemcpyHostToDevice ) );
    
    //  iterate through block columns
    for( int i = 0; i < n; i += PANEL_SIZE ) 
		{
        int h = n - i;
        int w = h < PANEL_SIZE ? h : PANEL_SIZE;
        if( i > 0 )
			{
			cublasSsyrk( 'L', 'N', w, PANEL_SIZE, -1, &A[i+(i-PANEL_SIZE)*lda], lda, 1, &A[i*(lda+1)], lda );
			SAFECALL( cublasGetError( ) );

			cublasSgemm( 'N', 'T', h-w, w, PANEL_SIZE, -1, &A[i+PANEL_SIZE+(i-PANEL_SIZE)*lda], lda, &A[i+(i-PANEL_SIZE)*lda], lda, 1, &A[i+PANEL_SIZE+i*lda], lda );
			SAFECALL( cublasGetError( ) );

			SAFECALL( cudaMemcpy2D( &cpu_A[i*(lda+1)], lda*sizeof(float), &A[i*(lda+1)], lda*sizeof(float), h*sizeof(float), w, cudaMemcpyDeviceToHost ) );
            
			if( h > PANEL_SIZE ) 	
				{
				cublasSsyrk( 'L', 'N', h-PANEL_SIZE, PANEL_SIZE, -1, &A[i+PANEL_SIZE+(i-PANEL_SIZE)*lda], lda, 1, &A[(i+PANEL_SIZE)*(lda+1)], lda );
				SAFECALL( cublasGetError( ) );
				}
			}
        
		choleskyDecompositionCPU(w, &cpu_A[i*(lda+1)], lda);
		if( h > PANEL_SIZE ) 
			{
	  		solveCPU(h - PANEL_SIZE, PANEL_SIZE, &cpu_A[i*(lda+1)], lda, &cpu_A[i+PANEL_SIZE+i*lda], lda );
	  		SAFECALL( cudaMemcpy2D( &A[i*(lda+1)], lda*sizeof(float), &cpu_A[i*(lda+1)], lda*sizeof(float), h*sizeof(float), w, cudaMemcpyHostToDevice ) );
			}
		}
} // choleskyDecompositionGPU	


void invert(float * A, int lda, int n) {
  fprintf(stderr,"inversion started");
/*#ifdef VERIFY	
	float * xcopy =new float[n*n];
	printf("\n hello \n");
	  for (int i = 0; i < n; i++) {
	    for (int j = 0; j < n; j++) {
	      xcopy[i*n+j] = A[i*lda+j];
	    }
	  }
		printf("\n hello \n");
	// mathdisp(xcopy,n,n);
#endif */

    volatile clock_t gputime, gputime0;
    gputime=clock();
    gputime0 = gputime;

    float * A_d;
    int m = (n+31)&~31;
	
    SAFECALL(cudaMalloc((void**) &A_d, lda*m*2*sizeof(float)));

    choleskyDecompositionGPU(n, A,  A_d, lda);
	
	SAFECALL(cudaMemcpy(A_d, A, lda*n*sizeof(float),cudaMemcpyHostToDevice));
        
	// create identity matrix
	float *B_h = new float[n*lda];
	
	memset(B_h, 0, lda*n*sizeof(float));
	for (int i=0;i<n;i++){
	  B_h[i*lda+i] = 1;
	}

	// solve
	float *B = &A_d[lda*m];
	//	SAFECALL(cudaMalloc((void**) &B, lda*n*sizeof(float)));
	SAFECALL(cudaMemcpy(B, B_h, lda*n*sizeof(float), cudaMemcpyHostToDevice));
	cublasStrsm ('L', 'L', 'N', 'N', n, n, 1.0f, A_d, lda, B, lda);
	cudaThreadSynchronize();

	// solve
	cublasStrsm ('L', 'L', 'T', 'N', n, n, 1.0f, A_d, lda, B, lda);
	cudaThreadSynchronize();
	SAFECALL(cudaMemcpy(A, B, lda*n*sizeof(float), cudaMemcpyDeviceToHost));
	//mathdisp(A,n,n);        

	//	delete [] A;
	SAFECALL(cudaFree(A_d));
	delete [] B_h;

	// now A = inverse(spd_matrix)
	gputime=clock()-gputime;fprintf(stderr, " %7.1f ms ",gputime/1.e3f);

   // fprintf(stderr, " %7.2f Glops", 1e-3*(2.0+3.0+3.0)*n*n*n/3.0/gputime);
    /*#ifdef VERIFY	
	// let's verify that
	float error=0.0;
	float *I = new float[n*n];

	// multiply inverse*xcopy, should be Identity matrix
	for (int k = 0; k < n; k++) {
	  for (int j = 0; j < n; j++) {
	    float sum = 0;
	    for (int i = 0; i < n; i++) {
	      sum += A[j*lda+i]*xcopy[i*n+k];
	    }
	    I[k*n+j] = sum;
	    if (j!=k) {
	      error += sum * sum;
	    } else {
	      error += (1.0-sum) * (1.0-sum);
	    }
	  }
	}
        fprintf(stderr, " %6.2f SSE", error);
#endif	
	fprintf(stderr," done!\n"); */
} // invert 
