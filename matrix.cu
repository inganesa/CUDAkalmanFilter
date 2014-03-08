#include <stdio.h>
#include <math.h>
#include <stdlib.h>

//Global Variable Declaration
float* p;

//Memory Allocation Function
void memAlloc(float **data_ptr, int dim_x, int dim_y)
{
	float *data;
	data = (float *) malloc(sizeof(float *) * dim_x * dim_y);
	*data_ptr = data;
}
 
void cleanp()
{
if(p)
	free(p);
}
 
/* ----------------------------------------------------
        main method for Cholesky decomposition.

        input         n  size of matrix
        input/output  a  Symmetric positive def. matrix
        output        p  vector of resulting diag of a
  ----------------------------------------------------- */
int choldc1(int n, float* a) 
{
	int i,j,k;
	float sum;
	for (i = 0; i < n; i++)
		{
		for (j = i; j < n; j++) 
			{
			sum = a[i * n + j];
			for (k = i - 1; k >= 0; k--) 
				{
				sum -= a[i * n + k] * a[j * n + k];
				}
			if (i == j) 
				{
                if (sum <= 0) 
					{
					printf(" S is not positive definite!\n");
					return 0;
					}
                p[i] = sqrt(sum);
				}
			else 
				{
                a[j * n + i] = sum / p[i];
				}
			}
		}
	return 1;
}

/* -----------------------------------------------------
         Inverse of Cholesky decomposition.

         input    n  size of matrix
         input    A  Symmetric positive def. matrix
         output   a  inverse of lower deomposed matrix
         uses        choldc1(int,MAT,VEC)         
   ----------------------------------------------------- */
int choldcsl(int n, float* A, float* a) 
{
	int i,j,k; float sum;
	int success;
	for (i = 0; i < n; i++) 
	for (j = 0; j < n; j++) 
		a[i * n + j] = A[i * n + j];
	success = choldc1(n, a);
	if (success == 0)
		return 0;
	for (i = 0; i < n; i++) 
		{
		a[i * n + i] = 1 / p[i];
		for (j = i + 1; j < n; j++) 
			{
			sum = 0;
			for (k = i; k < j; k++) 
				{
				sum -= a[j * n + k] * a[k * n + i];
				}
			a[j * n + i] = sum / p[j];
			}
		}
	return 1;
}
 
/* ---------------------------------------------------
        Matrix inverse using Cholesky decomposition

        input    n  size of matrix
        input	  A  Symmetric positive def. matrix
        output   a  inverse of A
        uses        choldc1(MAT, VEC)
   --------------------------------------------------- */
int inverse(int n, float* A, float* a) 
{
	int i,j,k,success;
	//temp memory allocation for p
	memAlloc(&p,n,n);
	//printf("\n memory allocation done \n");
	success = choldcsl(n,A,a);
	if( success == 0)
		{
		//cleanp();
		return 0;
		}
	for (i = 0; i < n; i++) 
		{
		for (j = i + 1; j < n; j++) 
			{
			a[i * n + j] = 0.0;
			}
		}
	for (i = 0; i < n; i++) 
		{
		a[i * n + i] *= a[i * n + i];
		for (k = i + 1; k < n; k++) 
			{
			a[i * n + i] += a[k * n + i] * a[k * n + i];
			}
		for (j = i + 1; j < n; j++) 
			{
			for (k = j; k < n; k++) 
				{
				a[i * n + j] += a[k * n + i] * a[k * n + j];
				}
			}
		}
	for (i = 0; i < n; i++) 
		{
		for (j = 0; j < i; j++) 
			{
			a[i * n + j] = a[j * n + i];
			}
		}
	//cleanp();
	return 1;
}
	
//Inversion Complete
//Other Matrix operations

//Addition
void add(float* C, float* A, float* B, int h, int w)
{
	int i,j;
	for (i = 0; i < h; i++)
		{
		for(j = 0; j < w ;j++)
			{
			C[i * w +j] = A[i * w + j] + B[i * w + j];
			}
		}
}

//subtraction
void sub(float* C, float* A, float* B, int h, int w)
{
	int i,j;
	for (i = 0; i < h; i++)
		{
		for(j = 0; j < w ;j++)
			{
			C[i * w + j] = A[i * w + j] - B[i * w + j];
			}
		}
}
 
//Multiplication
void mult(float* C, float* A, float* B, int ha, int wa, int hb, int wb)
{
	int i,j,k;
	float sum,a = 0,b = 0;
   	for (i = 0; i < ha; i++)
		{
		for(j = 0; j < wb ;j++)
			{
			sum = 0;
			for(k = 0; k < wa; k++)
				{
				a = A[i * wa + k];
				b = B[k * wb + j];
				sum += a * b;
				}
			C[i * wb + j] = sum;
			}
		}
}
 
//Transpose
void transpose(float* B, float* A,int h, int w)
{
	int i,j;
	for (i = 0; i < h; i++)
		{
		for(j = 0; j < w ;j++)
			{
			B[j * h + i] = A[i * w + j];
			}
		}
}

 
//print the matrix
void matPrint(float *A, int h, int w)
{
	int i,j;
	for(i = 0;i < h;i++)
		{
		for(j = 0;j < w;j++)
			{
			printf("%f ", A[i * w + j]);
			}
		printf("\n");}
}

//Matrix Copy
void matcopy(float *B, float *A, int h, int w)
{
	int i;
	for(i = 0;i < (h*w);i++)
		B[i] = A[i];
}
 
// generating L
void generateL(float *L, int n)
{
	int i,j;
	srand(1);
	for (i = 0; i < n; i++)
		{
		for(j = 0; j < n ;j++)
			{
			if(j <= i)
				L[i*n + j]= (rand() % 10) + 1;
			else
				L[i*n + j] = 0;
			}
		}
}

//Random Initialize
void RandomInit(float* data, int n1, int n2)
{   
	srand(1);
    for (int i = 0; i < (n1*n2); ++i)
        data[i] = (rand() % 10) + 1;
}

//Ideintity Matrix Generation
void Identity(float *data, int n)
{
	for (int i = 0; i < (n*n); i=i+1)
		{
		if((i%(n+1))==0)
			data[i] = 1;
		else
			data[i] = 0;
		}        
}

void Initialize(float *X,float *P,float *F,float *Z,float *H,float *E,float *I,float *Ht,float *Ft,float *s, int ns, int no)
{
	RandomInit(X, ns, 1);
	RandomInit(Z, no, 1);
	RandomInit(H, no, ns);

	transpose(Ht,H,no,ns);
	//printf("\n Transpose of H successful\n");

	float *P1;
	float *P2;
	memAlloc(&P1,ns,ns);
	memAlloc(&P2,ns,ns);
	generateL(P1,ns);
	transpose(P2,P1,ns,ns);
	mult(P,P1,P2,ns,ns,ns,ns);
	if(P1)
		free(P1);
	if(P2)
		free(P2);

	float *F1;
	float *F2;
	memAlloc(&F1,ns,ns);
	memAlloc(&F2,ns,ns);
	generateL(F1,ns);
	transpose(F2,F1,ns,ns);
	mult(F,F1,F2,ns,ns,ns,ns);
	if(F1)
		free(F1);
	if(F2)
		free(F2);
	
	transpose(Ft,F,ns,ns);
	//printf("\n Transpose of F successful\n");

	float *E1;
	float *E2;
	memAlloc(&E1,no,no);
	memAlloc(&E2,no,no);
	generateL(E1,no);
	transpose(E2,E1,no,no);
	mult(E,E1,E2,no,no,no,no);
	if(E1)
		free(E1);
	if(E2)
		free(E2);
	
	float *s1;
	float *s2;
	memAlloc(&s1,no,no);
	memAlloc(&s2,no,no);
	generateL(s1,no);
	transpose(s2,s1,no,no);
	mult(s,s1,s2,no,no,no,no);
	if(s1)
		free(s1);
	if(s2)
		free(s2);
		
	Identity(I, ns);
}





