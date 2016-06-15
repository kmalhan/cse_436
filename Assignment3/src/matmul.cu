/*
 *	Assignment 3 (CSE436)
 *	Kazumi Malhan
 *	06/20/2016
 *
 *	Rectangular matrix multiplication
 *	A[N][N] * B[N][N] = C[N][N]
 */

// NOTE: This is version v0.4
// NOTE: This code should be executed on yoko.secs.oakland.edu
// REVIEW: Delete all debug printf if exists

/* Include Files */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/timeb.h>
#include <string.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

/* read timer in second */
double read_timer() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time + (double) tm.millitm / 1000.0;
}

/* read timer in ms */
double read_timer_ms() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time * 1000.0 + (double) tm.millitm;
}

/* Macro */
#define REAL float
#define BLOCK_SIZE 16

/* Matrix Initialization */
void init(int M, int N, REAL * A) {
    int i, j;

    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            A[i*N+j] = (REAL) drand48();
        }
    }
}

/* Maximum Error Calculation */
double maxerror(int M, int N, REAL * A, REAL *B) {
    int i, j;
    double error = 0.0;

    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            double diff = (A[i*N+j] - B[i*N+j]) / A[i*N+j];
            if (diff < 0)
                diff = -diff;
            if (diff > error)
                error = diff;
        }
    }
    return error;
}

/* Function Prototype (Host) */
void matmul_base(int N, REAL *A, REAL * B, REAL *C);
void matmul_openmp(int N, REAL *A, REAL *B, REAL *C, int num_tasks);
void matmul_cuda_v1_vanilla(int N, REAL *A, REAL *B, REAL *C);
void matmul_cuda_v1_shmem(int N, REAL *A, REAL *B, REAL *C);
void matmul_cuda_v1_cublas(int N, REAL *A, REAL *B, REAL *C);

/* Function Prototype (Device) */
__global__ void matmul_global_kernel(int N, REAL* A, REAL* B, REAL* C);
__global__ void matmul_shared_kernel(int N, REAL* A, REAL* B, REAL* C);

/*
 *	Host Main Function
 *	To compile: nvcc –Xcompiler -fopenmp matmul.cu -lpthread -lcublas -o matmul
 *	To run: ./matmul N num_tasks
 */
int main(int argc, char *argv[]) {
    int N;
    int num_tasks = 5; /* 5 is default number of tasks */
    double elapsed_base, elapsed_openmp, elapsed_cuda_v1, elapsed_cuda_v2, elapsed_cuda_v3; /* for timing */
    if (argc < 2) {
        fprintf(stderr, "Usage: matmul <n> [<#tasks(%d)>]\n", num_tasks);
        exit(1);
    }
    N = atoi(argv[1]);
    if (argc > 2) num_tasks = atoi(argv[2]);
    // modified to incorporate 5 array (originally 4)
    REAL * heap_buffer = (REAL*)malloc(sizeof(REAL)*N*N*7); /* we use 5 matrix in this example */
    /* below is a cast from memory buffer to a 2-d row-major array */
    REAL *A = heap_buffer;
    REAL *B = &heap_buffer[N*N];
    REAL *C_base = &heap_buffer[2*N*N];
    REAL *C_openmp = &heap_buffer[3*N*N];
    REAL *C_cuda_v1 = &heap_buffer[4*N*N];		// added
    REAL *C_cuda_v2 = &heap_buffer[5*N*N];		// added
    REAL *C_cuda_v3 = &heap_buffer[6*N*N];		// added
    // REVIEW: If it is necessary to have separate matrix for all cuda function

    srand48((1 << 12));
    init(N, N, A);
    init(N, N, B);

    /* example run */
    elapsed_base = read_timer();
    matmul_base(N, A, B, C_base);
    elapsed_base = (read_timer() - elapsed_base);

    elapsed_openmp = read_timer();
    matmul_openmp(N, A, B, C_openmp, num_tasks);
    elapsed_openmp = (read_timer() - elapsed_openmp);

    /* call and timing for the three CUDA versions */
    /* there are three devices you can use on gpu.secs.oakland.edu, 0, 2, 3.
     * 1 is a graphical card with less computation capability.
     */
    cudaSetDevice(0);

    /* call and time for matmul_cuda_v1_vanilla(int N, REAL *A, REAL *B, REAL *C); */
    elapsed_cuda_v1 = read_timer();
    matmul_cuda_v1_vanilla(N, A, B, C_cuda_v1);
    elapsed_cuda_v1 = (read_timer() - elapsed_cuda_v1);

    /* call and time for matmul_cuda_v1_shmem(int N, REAL *A, REAL *B, REAL *C); */
    elapsed_cuda_v2 = read_timer();
    matmul_cuda_v1_shmem(N, A, B, C_cuda_v2);
    elapsed_cuda_v2 = (read_timer() - elapsed_cuda_v2);

    // TODO: Test of cublas run
    /* call and time for matmul_cuda_v1_cublas(int N, REAL *A, REAL *B, REAL *C); */
    elapsed_cuda_v3 = read_timer();
    matmul_cuda_v1_cublas(N, A, B, C_cuda_v3);
    elapsed_cuda_v3 = (read_timer() - elapsed_cuda_v3);

    printf("======================================================================================================\n");
    printf("Matrix Multiplication: A[M][K] * B[k][N] = C[M][N], M=K=N=%d, %d threads/tasks\n", N, num_tasks);
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Performance:\t\tRuntime (ms)\t MFLOPS \t\tError (compared to base)\n");
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("matmul_base:\t\t%4f\t%4f \t\t%g\n", elapsed_base * 1.0e3, ((((2.0 * N) * N) * N) / (1.0e6 * elapsed_base)), maxerror(N, N, C_base, C_base));
    printf("matmul_openmp:\t\t%4f\t%4f \t\t%g\n", elapsed_openmp * 1.0e3, ((((2.0 * N) * N) * N) / (1.0e6 * elapsed_openmp)), maxerror(N, N, C_base, C_openmp));

    /* put other printf statements for outputing results for GPU execution */
    printf("matmul_global:\t\t%4f\t%4f \t\t%g\n", elapsed_cuda_v1  * 1.0e3, ((((2.0 * N) * N) * N) / (1.0e6 * elapsed_cuda_v1 )), maxerror(N, N, C_base, C_cuda_v1));
    printf("matmul_shared:\t\t%4f\t%4f \t\t%g\n", elapsed_cuda_v2  * 1.0e3, ((((2.0 * N) * N) * N) / (1.0e6 * elapsed_cuda_v2 )), maxerror(N, N, C_base, C_cuda_v2));
    printf("matmul_cublas\t\t%4f\t%4f \t\t%g\n", elapsed_cuda_v3  * 1.0e3, ((((2.0 * N) * N) * N) / (1.0e6 * elapsed_cuda_v3 )), maxerror(N, N, C_base, C_cuda_v3));

    free(heap_buffer);

    /* Reset device and exit */
    cudaDeviceReset();
    return 0;
}

/* Serial implementation */
void matmul_base(int N, REAL *A, REAL * B, REAL *C) {
    int i, j, k;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            REAL temp = 0.0;
            for (k = 0; k < N; k++) {
                temp += A[i*N+k] * B[k*N+j];
            }
            C[i*N+j] = temp;
        }
    }
}

/* OpenMP inplementation */
void matmul_openmp(int N, REAL *A, REAL *B, REAL *C, int num_tasks) {
    int i, j, k;
#pragma omp parallel for shared(N,A,B,C,num_tasks) private(i,j,k) num_threads(num_tasks)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            REAL temp = 0.0;
            for (k = 0; k < N; k++) {
                temp += A[i*N+k] * B[k*N+j];
            }
            C[i*N+j] = temp;
        }
    }
}

/*
 * call to kernel that uses GPU global memory
 */
void matmul_cuda_v1_vanilla(int N, REAL *A, REAL *B, REAL *C) {

  // Calculate allocation size for GPU memory
  size_t size = sizeof(REAL)*N*N;

  // Allocate GPU memory for A
  REAL* d_A = NULL;
  cudaMalloc((void**)&d_A, size);

  // Allocate GPU memory for B
  REAL* d_B = NULL;
  cudaMalloc((void**)&d_B, size);

  // Allocate GPU memory for C
  REAL* d_C = NULL;
  cudaMalloc((void**)&d_C, size);

  // Copy A and B from Host to Device
  cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

  // Launch matrix multiplication with GPU global memory kernel
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid( N/dimBlock.x, N/dimBlock.y );
  matmul_global_kernel <<< dimGrid, dimBlock >>>(N, d_A, d_B, d_C);

  // Read Result back to CPU memory
  cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

/*
 * call to kernel that use GPU shared memory
 */
void matmul_cuda_v1_shmem(int N, REAL *A, REAL *B, REAL *C) {

  // Calculate allocation size for GPU memory
  size_t size = sizeof(REAL)*N*N;

  // Allocate GPU memory for A
  REAL* d_A = NULL;
  cudaMalloc((void**)&d_A, size);

  // Allocate GPU memory for B
  REAL* d_B = NULL;
  cudaMalloc((void**)&d_B, size);

  // Allocate GPU memory for C
  REAL* d_C = NULL;
  cudaMalloc((void**)&d_C, size);

  // Copy A and B from Host to Device
  cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

  // Launch matrix multiplication with GPU global memory kernel
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid( N/dimBlock.x, N/dimBlock.y );
  matmul_shared_kernel <<< dimGrid, dimBlock >>>(N, d_A, d_B, d_C);

  // Read Result back to CPU memory
  cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

// TODO: Complete function using cublas
/*
 * call to sgemm of cublas library
 */
void matmul_cuda_v1_cublas(int N, REAL *A, REAL *B, REAL *C) {

  // Size of matrix
  size_t size = sizeof(REAL)*N*N;

  // Allocate GPU memory for matrix A, B, C
  REAL* d_A;
  cudaMalloc((void**)&d_A, size);
  REAL* d_B;
  cudaMalloc((void**)&d_B, size);
  REAL* d_C;
  cudaMalloc((void**)&d_C, size);

  // Initialize cuBLAS handle
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Copy data of matrix A and B
  cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

  // REVIEW: Check if this operation is correct!!
  // use cublasSgemm as data is REAL (float)
  REAL alpha = 1.0f;
  REAL beta = 1.0f;
  // (handle, transa, transb, m, n, k, *alpha, *A, lda, *B, ldb, *beta, *C, ldc)
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);

  // Copy the result back to CPU
  cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

  // Free GPU memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

/*
 * CUDA kernel function with GPU global memory
 */
__global__ void matmul_global_kernel(int N, REAL* A, REAL* B, REAL* C){

  // Local variable
  REAL temp = 0.0;
  int i;

  // Calculate thread location in terms of row and col
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Perform calculation
  for (i=0; i<N; i++)
    temp += A[row*N+i] * B[i*N+col];

    // Write result to C
    C[row*N+col] = temp;
}

/*
 * CUDA kernel function with GPU shared memory
 */
__global__ void matmul_shared_kernel(int N, REAL* A, REAL* B, REAL* C){

  // Local variable
  REAL temp = 0.0;
  int i, j;

  // Determine thread location
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Determine row and column
  int row = blockIdx.y * BLOCK_SIZE + ty;
  int col = blockIdx.x * BLOCK_SIZE + tx;

  // Shared memory to store sub-matrix
  __shared__ REAL Asub[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ REAL Bsub[BLOCK_SIZE][BLOCK_SIZE];

  // Copy matrix into shared memory sub-matrix REVIEW: <= is correct?
  // When you create sub-matrix, it is important to make sure
  // threads within matrix is performing the calculation
  for (i=0; i<=(N/BLOCK_SIZE); i++) {

      // Matrix A
      if ( row<N && (i*BLOCK_SIZE+tx)<N )
        Asub[ty][tx] = A[row*N+(i*BLOCK_SIZE+tx)];
      else
        Asub[ty][tx] = 0.0;

      // Matrix B
      if ( col<N && (i*BLOCK_SIZE+ty)<N )
        Bsub[ty][tx] = B[(i*BLOCK_SIZE+ty)*N+col];
      else
        Bsub[ty][tx] = 0.0;

      // Synchronize all threads (make sure sub-matrix is developed)
      __syncthreads();

      // Calculate each element of d_C
      for (j=0; j<BLOCK_SIZE; j++)
        temp += Asub[ty][j] * Bsub[j][tx];
      // Synchronize all threads (make sure calculateion is ended)
      __syncthreads();
  }

  // If threads are within elements of d_C, put result to d_C
  if ( row<N && col<N )
    C[row*N+col] = temp;
}
