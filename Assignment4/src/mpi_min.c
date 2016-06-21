/*
 * Find the min from an array, and output the value
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/timeb.h>
#include <string.h>
#include "mpi.h"

#define REAL float

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

void init(int N, REAL *A) {
    int i;

    for (i = 0; i < N; i++) {
        A[i] = (REAL) drand48();
    }
}

/*
 * N: The size of the REAL array A
 * A: The REAL array pointer
 * min: the min output
 */
int main(int argc, char *argv[]) {
    int N = 1024 * 512; /* the size of the global REAL array A */
    int local_N; /* local portion of the array each process to work on */
    REAL *A; /* the pointer to the global array A */
    REAL *local_A; /* the pointer to the local buffer by each process */
    int i;
    REAL* temp;
    
    /* results */
    REAL min;

    /* for timing */
    double elapsed;
    int numprocs, myrank;
    int root = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Status status;  
  
    if (myrank == 0) {
        fprintf(stderr, "Usage: min [<array size N, default: %d]\n", N);
        fprintf(stderr, "\tSize N should be dividable by num MPI processes]\n", N);
    }
    if (argc > 1) N = atoi(argv[1]);

    /*compute the local_N */
    local_N = N / numprocs;

    /* init the buffer, you may need to change this for your algorithm */
    if (myrank == 0) {
        A = (REAL *) malloc(sizeof(REAL) * N);
        srand48((1 << 12));
        init(N, A);
        local_A = A;
        temp = (REAL *) malloc(sizeof(REAL) * numprocs);
        for (i=0; i<N; i++){
          printf("values: %f \n", A[i]);
        }
    } else {
        local_A = (REAL *) malloc(sizeof(REAL) * local_N);
    }
    
    /***** Version 2 code (using Send and Recv) *****/
    /* Start timer */
    if (myrank == 0) elapsed = read_timer();
    
    /* Step1: Rank 0 distrubutes the array into workers */ 
    if (myrank == 0) {
      for (i=1; i<numprocs; i++){
          REAL *send_buffer = &A[local_N*i];
          MPI_Send(send_buffer, local_N, MPI_FLOAT, i, 1234, MPI_COMM_WORLD);
      }
    } else {
      MPI_Recv(local_A, local_N, MPI_FLOAT, 0, 1234, MPI_COMM_WORLD, &status);
    }

    /* Step2: Now we all compute the local min in a for loop */
    min = *local_A;
    for (i=1; i<local_N; i++){
        if (min > local_A[i])
            min = local_A[i];
    }
    printf("I am %d, and min value is %f\n", myrank, min);

    /* Step3: Send the local min to rank 0 */
    if (myrank == 0){
      for (i=1; i<numprocs; i++){
          MPI_Recv(&temp[i], 1, MPI_FLOAT, i, 4321, MPI_COMM_WORLD, &status); 
      }
    } else {
      MPI_Send(&min, 1, MPI_FLOAT, 0, 4321, MPI_COMM_WORLD);
    }
    
    /* Step4: Rank 0 compute the final min */
    if (myrank == 0) {
      for (i=1; i<numprocs; i++){
          if (min > temp[i])
                min = temp[i];
      }
    }
    /**** End of version 2 ****/

    /* your implementation here for one of the version */
//    printf("Hello World: %d of %d\n", myrank, numprocs);

    if (myrank == 0) {
        elapsed = (read_timer() - elapsed);
        printf("======================================================================================================\n");
        printf("Finding min of array of %d floats using %d MPI processes, implemented using MPI_Scatter/Reduce calls\n", N, numprocs);
        printf("Result: min: %f\n", min);
        printf("Executime Time: %f seconds\n", elapsed);
    }

    free(local_A);
    MPI_Finalize();
    return 0;
}
