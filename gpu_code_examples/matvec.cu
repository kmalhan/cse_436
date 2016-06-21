/**
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

/**
 * CUDA Kernel Device code
 *
 * Computes matrix vector multiplication: A[N][N] * B[N] = C[N]
 *
 * 
 */

void matvec(float * A, float * B, float * C, int N) {
    int i, j;
    for (i=0; i<N; i++) {
      float temp = 0.0;
      for (j=0; j<N; j++)
        temp += A[i*N+j] * B[j];

      C[i] = temp; 
    }
}

__global__ void
matvec_kernel(float * A, float * B, float * C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j;

    if (i < N) {
      float temp = 0.0;
      for (j=0; j<N; j++)
        temp += A[i*N+j] * B[j];

      C[i] = temp;
    }
}

/**
 * N =1024 
 * 4 blocks, 256 threads/per block 
 */

__global__ void
matvec_kernel_shared(float * A, float * B, float * C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x; /* 0 - 1023 */
    int j;

    extern __shared__ float B_shared[]; 
    B_shared[i] = B[i];
    /* for block 0: 0-255 are filled */
    /* for block 1: 256-511 are filled */
    /* for block 2: 512-767 are filled */
    /* for block 3: 768 - 1023 are filled */

    B_shared[(i+256)%1024] = B[(i+256)%1024];
    B_shared[(i+512)%1024] = B[(i+512)%1024];
    B_shared[(i+768)%1024] = B[(i+768)%1024];

    __syncthreads();

    if (i < N) {
      float temp = 0.0;
      for (j=0; j<N; j++)
        temp += A[i*N+j] * B_shared[j];

      C[i] = temp;
    }
}

__global__ void
matvec_kernel_shared_general(float * A, float * B, float * C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x; /* 0 - 1023 */
    int j;

    extern __shared__ float B_shared[]; 
    int k;
    for (k=0; k<gridDim.x; k++) {
    	B_shared[(threadIdx.x + k*blockDim.x)%N] = B[(threadIdx.x + k*blockDim.x)%N];
    }

    __syncthreads();

    if (i < N) {
      float temp = 0.0;
      for (j=0; j<N; j++)
        temp += A[i*N+j] * B_shared[j];

      C[i] = temp;
    }
}

void vectorAdd_sequential(const float *A, const float *B, float *C, int numElements)
{
    int i;
    for (i=0; i < numElements; i++)
    {
        C[i] = A[i] + B[i];
    }
}

__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

/**
 * Host main routine
 */
int
main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int numElements = 1024;
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host input vector A
    float *h_A = (float *)malloc(size*size);

    // Allocate the host input vector B
    float *h_B = (float *)malloc(size);

    // Allocate the host output vector C
    float *h_C = (float *)malloc(size);

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // Allocate the device input vector A
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size*size);

    // Allocate the device input vector B
    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);

    // Allocate the device output vector C
    float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size*size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = 4; //(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    matvec_kernel<<<blocksPerGrid, threadsPerBlock, size>>>(d_A, d_B, d_C, numElements);
    err = cudaGetLastError();

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // Free device global memory
    err = cudaFree(d_A);
    err = cudaFree(d_B);
    err = cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();

    printf("Done\n");
    return 0;
}

