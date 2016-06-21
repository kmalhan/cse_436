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

    /* results */
    REAL min;

    /* for timing */
    double elapsed;
    int numprocs, myrank;
    int root = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

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
    } else {
        local_A = (REAL *) malloc(sizeof(REAL) * N);
    }

    if (myrank == 0) elapsed = read_timer();


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
