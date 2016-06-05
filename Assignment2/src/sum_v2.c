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
 * To run: ./sum N num_tasks
 */
int main(int argc, char *argv[]) {
    int N = VECTOR_LENGTH;
    int num_tasks = 4;
    double elapsed_serial, elapsed_para, elapsed_para_for; /* for timing */
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
	
	/* Parallel Run */
    elapsed_para = read_timer();
    result = sum_omp_parallel(N, A, num_tasks);
    elapsed_para = (read_timer() - elapsed_para);
	
	/* Parallel For Run */
    elapsed_para_for = read_timer();
    result = sum_omp_parallel_for(N, A, num_tasks);
    elapsed_para_for = (read_timer() - elapsed_para_for);
	
    /* you should add the call to each function and time the execution */
    printf("======================================================================================================\n");
    printf("\tSum %d numbers with %d tasks\n", N, num_tasks);
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Performance:\t\tRuntime (ms)\t MFLOPS \n");
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Sum Serial:\t\t\t%4f\t%4f\n", 		elapsed_serial * 1.0e3, 	2*N / (1.0e6 * elapsed_serial));
	  printf("Sum Parallel:\t\t\t%4f\t%4f\n", 	elapsed_para * 1.0e3, 		2*N / (1.0e6 * elapsed_para));
	  printf("Sum Parallel For:\t\t%4f\t%4f\n", elapsed_para_for * 1.0e3, 	2*N / (1.0e6 * elapsed_para_for));
    
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
REAL sum_omp_parallel (int N, REAL *A, int num_tasks) {
	REAL result = 0.0;
  int partial_result[num_tasks];
	int t;

	/* Determine if task can be evenly distrubutable */
	int each_task = N / num_tasks;
	int leftover = N - (each_task * num_tasks);
  
  #pragma omp parallel shared (N, A, num_tasks, leftover) num_threads(num_tasks)
	{
		int i, tid, istart, iend;
    REAL temp;
		tid = omp_get_thread_num();	
		istart = tid * (N / num_tasks);
		iend = (tid + 1) * (N / num_tasks);
		
		for (i = istart; i < iend; ++i) {
      temp += A[i];
		}
		
		/* Take care left over */
		if (tid < leftover) {
      temp += A[N - tid - 1];
		}
    partial_result[tid] = temp;
    
	} // end of parallel
 
  /* Add result together */ 
  for(t=0; t<num_tasks; t++)
    result += partial_result[t];

  return result;
}

/* Parallel For Implemenration */
REAL sum_omp_parallel_for (int N, REAL *A, int num_tasks) {
    int i;
    REAL result = 0.0;
	# pragma omp parallel shared (N, A, result) private (i) num_threads(num_tasks)
	{
		# pragma omp for schedule(runtime) reduction (+:result) nowait
			for (i = 0; i < N; ++i) {
				result += A[i];
      }
	} // end of parallel
    return result;
}

