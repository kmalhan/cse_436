/*
 * Sum of A[N]
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>

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

#define REAL float
#define VECTOR_LENGTH 102400

/* initialize a vector with random floating point numbers */
void init(REAL *A, int N) {
    int i;
    for (i = 0; i < N; i++) {
        A[i] = (double) drand48();
    }
}

REAL sum (int N, REAL *A);
REAL sum_omp_parallel (int N, REAL *A, int num_tasks);
REAL sum_omp_parallel_for (int N, REAL *A, int num_tasks);

int main(int argc, char *argv[]) {
    int N = VECTOR_LENGTH;
    int num_tasks = 4;
    double elapsed; /* for timing */
    if (argc < 3) {
        fprintf(stderr, "Usage: sum [<N(%d)>] [<#tasks(%d)>]\n", N,num_tasks);
        fprintf(stderr, "\t Example: ./sum %d %d\n", N,num_tasks);
    } else {
    	N = atoi(argv[1]);
    	num_tasks = atoi(argv[2]);
    }

    REAL *A = (REAL*)malloc(sizeof(REAL)*N);

    srand48((1 << 12));
    init(A, N);
    /* example run */
    elapsed = read_timer();
    REAL result = sum(N, A);
    elapsed = (read_timer() - elapsed);

    /* more runs */

    /* you should add the call to each function and time the execution */
    printf("======================================================================================================\n");
    printf("\tSum %d numbers with %d tasks\n", N, num_tasks);
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Performance:\t\tRuntime (ms)\t MFLOPS \n");
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Sum:\t\t\t%4f\t%4f\n", elapsed * 1.0e3, 2*N / (1.0e6 * elapsed));
    free(A);
    return 0;
}

REAL sum(int N, REAL *A) {
    int i;
    REAL result = 0.0;
    for (i = 0; i < N; ++i)
        result += A[i];
    return result;
}
