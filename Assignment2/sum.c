/*
 * Assignment 2 (CSE436)
 * Kazumi Malhan
 * 06/08/2016
 *
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

/* Function Prototypes */
REAL sum (int N, REAL *A);
REAL sum_omp_parallel (int N, REAL *A, int num_tasks);
REAL sum_omp_parallel_for (int N, REAL *A, int num_tasks);

/* 
 * To compile: gcc sum.c -fopenmp -o sum
 * To run: ./sum
 *
 */

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
    
	/* Serial Run */
    elapsed_serial = read_timer();
    REAL result = sum(N, A);
    elapsed_serial = (read_timer() - elapsed_serial);
	
	printf("Serial Result: %f\n", result); // debug
	
	/* Parallel Run */
    elapsed_para = read_timer();
    REAL result = sum_omp_parallel(N, A, num_tasks);
    elapsed_para = (read_timer() - elapsed_para);
	
	printf("Serial Result: %f\n", result); // debug
	
	/* Parallel For Run */
    elapsed_para_for = read_timer();
    REAL result = sum_omp_parallel_for(N, A, num_tasks);
    elapsed_para_for = (read_timer() - elapsed_para_for);
	
	printf("Serial Result: %f\n", result); // debug

    /* you should add the call to each function and time the execution */
    printf("======================================================================================================\n");
    printf("\tSum %d numbers with %d tasks\n", N, num_tasks);
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Performance:\t\tRuntime (ms)\t MFLOPS \n");
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Sum:\t\t\t%4f\t%4f\n", elapsed_serial * 1.0e3, 2*N / (1.0e6 * elapsed_serial));
    
	
	free(A);
    return 0;
}

/* Serial Implemenration */
REAL sum(int N, REAL *A) {
    int i;
    REAL result = 0.0;
    for (i = 0; i < N; ++i)
        result += A[i];
    return result;
}

/* Parallel Implemenration */
REAL sum(int N, REAL *A, int num_tasks) {
	REAL result = 0.0;
	#pragma omp parallel shared (N, A, num_tasks, result)
	{
		int i, tid, istart, iend;
		
		tid = omp_get_thread_num();
		istart = tid * (N / num_tasks);
		iend = (tid + 1) * (N / num_tasks);
		
		for (i = istart; i < iend; ++i) {
			#pragma omp atomic
			result += A[i];	/* Must be atomic */
		}
	} // end of parallel
    return result;
}

/* Parallel For Implemenration */
REAL sum(int N, REAL *A, int num_tasks) {
    int i;
    REAL result = 0.0;
	omp_set_num_threads(num_tasks);
	# pragma omp parallel shared (N, A, result) private (i)
	{
		# pragma omp for schedule(static) nowait
		{
			for (i = 0; i < N; ++i)
				#pragma omp atomic
				result += A[i];
		}
	} // end of parallel
    return result;
}

