/*
 * Assignment2 (CSE436)
 * Kazumi Malhan
 * 06/08/2016
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
#define VECTOR_LENGTH 512

/* initialize a vector with random floating point numbers */
void init(REAL A[], int N) {
    int i;
    for (i = 0; i < N; i++) {
		A[i] = (double) drand48();
    }
}

/* Function Prototypes */
void mm(int N, int K, int M, REAL * A, REAL * B, REAL * C);
void mm_parallel_row(int N, int K, int M, REAL * A, REAL * B, REAL * C, int num_tasks);
void mm_parallel_col(int N, int K, int M, REAL * A, REAL * B, REAL * C, int num_tasks);
void mm_parallel_rowcol(int N, int K, int M, REAL * A, REAL * B, REAL * C, int num_tasks);
void mm_parallel_for_row(int N, int K, int M, REAL * A, REAL * B, REAL * C, int num_tasks);
void mm_parallel_for_col(int N, int K, int M, REAL * A, REAL * B, REAL * C, int num_tasks);
void mm_parallel_for_rowcol(int N, int K, int M, REAL * A, REAL * B, REAL * C, int num_tasks);

/**
 * To compile: gcc mm.c -fopenmp -o mm
 * TO run: ./mm N M K num_tasks
 */
int main(int argc, char *argv[]) {
    int N = VECTOR_LENGTH;
    int M = N;
    int K = N;
    int num_tasks = 4;
    double elapsed; /* for timing */
    if (argc < 5) {
        fprintf(stderr, "Usage: mm [<N(%d)>] <K(%d) [<M(%d)>] [<#tasks(%d)>]\n", N,K,M,num_tasks);
        fprintf(stderr, "\t Example: ./mm %d %d %d %d\n", N,K,M,num_tasks);
    } else {
    	N = atoi(argv[1]);
    	K = atoi(argv[2]);
    	M = atoi(argv[3]);
    	num_tasks = atoi(argv[4]);
    }
    printf("\tC[%d][%d] = A[%d][%d] * B[%d][%d] with %d tasks\n", N, M, N, K, K, M, num_tasks);
    REAL * A = malloc(sizeof(REAL)*N*K);
    REAL * B = malloc(sizeof(REAL)*K*M);
    REAL * C = malloc(sizeof(REAL)*N*M);

    srand48((1 << 12));
    init(A, N*K);
    init(B, K*M);

    /* Serial program */
    double elapsed_mm = read_timer();
    mm(N, K, M, A, B, C);
    elapsed_mm  = (read_timer() - elapsed_mm);

	/* Parallel program */
    double elapsed_mm_parallel_row = read_timer();
    mm_parallel_row(N, K, M, A, B, C, num_tasks);
    elapsed_mm_parallel_row  = (read_timer() - elapsed_mm_parallel_row);

	double elapsed_mm_parallel_col = read_timer();
    mm_parallel_col(N, K, M, A, B, C, num_tasks);
    elapsed_mm_parallel_col  = (read_timer() - elapsed_mm_parallel_col);

	double elapsed_mm_parallel_rowcol = read_timer();
    mm_parallel_rowcol(N, K, M, A, B, C, num_tasks);
    elapsed_mm_parallel_rowcol  = (read_timer() - elapsed_mm_parallel_rowcol);

	/* Parallel for program */
	double elapsed_mm_parallel_for_row = read_timer();
   mm_parallel_for_row(N, K, M, A, B, C, num_tasks);
    elapsed_mm_parallel_for_row  = (read_timer() - elapsed_mm_parallel_for_row);

	double elapsed_mm_parallel_for_col = read_timer();
    mm_parallel_for_col(N, K, M, A, B, C, num_tasks);
    elapsed_mm_parallel_for_col  = (read_timer() - elapsed_mm_parallel_for_col);

	double elapsed_mm_parallel_for_rowcol = read_timer();
    mm_parallel_for_rowcol(N, K, M, A, B, C, num_tasks);
    elapsed_mm_parallel_for_rowcol  = (read_timer() - elapsed_mm_parallel_for_rowcol);

    /* you should add the call to each function and time the execution */
    printf("======================================================================================================\n");
    printf("\tC[%d][%d] = A[%d][%d] * B[%d][%d] with %d tasks\n", N, M, N, K, K, M, num_tasks);
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Performance:\t\t\t\tRuntime (ms)\t MFLOPS \n");
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("mm:\t\t\t\t%4f\t%4f\n",  elapsed_mm * 1.0e3, M*N*K / (1.0e6 *  elapsed_mm));
    printf("mm_parallel_row:\t\t%4f\t%4f\n",  elapsed_mm_parallel_row * 1.0e3, M*N*K / (1.0e6 *  elapsed_mm_parallel_row));
	printf("mm_parallel_col:\t\t%4f\t%4f\n",  elapsed_mm_parallel_col * 1.0e3, M*N*K / (1.0e6 *  elapsed_mm_parallel_col));
	printf("mm_parallel_rowcol:\t\t%4f\t%4f\n",  elapsed_mm_parallel_rowcol * 1.0e3, M*N*K / (1.0e6 *  elapsed_mm_parallel_rowcol));
	printf("mm_parallel_for_row:\t\t%4f\t%4f\n",  elapsed_mm_parallel_for_row * 1.0e3, M*N*K / (1.0e6 *  elapsed_mm_parallel_for_row));
	printf("mm_parallel_for_col:\t\t%4f\t%4f\n",  elapsed_mm_parallel_for_col * 1.0e3, M*N*K / (1.0e6 *  elapsed_mm_parallel_for_col));
	printf("mm_parallel_for_rowcol:\t\t%4f\t%4f\n",  elapsed_mm_parallel_for_rowcol * 1.0e3, M*N*K / (1.0e6 *  elapsed_mm_parallel_for_rowcol));

    free(A);
    free(B);
    free(C);
    return 0;
}

/* Serial */
void mm(int N, int K, int M, REAL * A, REAL * B, REAL * C) {
    int i, j, w;
    for (i=0; i<N; i++)
        for (j=0; j<M; j++) {
	    REAL temp = 0.0;
	    for (w=0; w<K; w++)
	        temp += A[i*K+w]*B[w*M+j];
	    C[i*M+j] = temp;
	}
}

/* Parallel Row */
void mm_parallel_row(int N, int K, int M, REAL * A, REAL * B, REAL * C, int num_tasks){
	int i, j, w;
	omp_set_num_threads(num_tasks);
	#pragma omp parallel shared (N, K, M, A, B, C, num_tasks) private (i, j, w)
	{
		int tid, istart, iend;
		tid = omp_get_thread_num();
		istart = tid * (N / num_tasks);
		iend = (tid + 1) * (N / num_tasks);

		for (i=istart; i<iend; i++) { /* decompose this loop */
			for (j=0; j<M; j++) {
				REAL temp = 0.0;
				for (w=0; w<K; w++)
					temp += A[i*K+w]*B[w*M+j];
				C[i*M+j] = temp;
			}
		}
	}/* end of parallel */
}

/* Parallel Column */
void mm_parallel_col(int N, int K, int M, REAL * A, REAL * B, REAL * C, int num_tasks){
	int i, j, w;
	omp_set_num_threads(num_tasks);
	#pragma omp parallel shared (N, K, M, A, B, C, num_tasks) private (i, j, w)
	{
		int tid, jstart, jend;
		tid = omp_get_thread_num();
		jstart = tid * (M / num_tasks);
		jend = (tid + 1) * (M / num_tasks);

		for (i=0; i<N; i++) {
			for (j=jstart; j<jend; j++) { /* decompose this loop */
				REAL temp = 0.0;
				for (w=0; w<K; w++)
					temp += A[i*K+w]*B[w*M+j];
				C[i*M+j] = temp;
			}
		}
	} /* end of parallel */
}

/* Parallel Row Column */
void mm_parallel_rowcol(int N, int K, int M, REAL * A, REAL * B, REAL * C, int num_tasks){
    int i, j, w;
    int task_r, task_c;
    /* Calculate amount of work for each thread */
    if (num_tasks == 1){
      task_r = 1;
      task_c = 1;
    } else {
      task_r = num_tasks / 2;
      task_c = num_tasks / task_r;
    }

    #pragma omp parallel shared (N, K, M, A, B, C, task_r, task_c) private (i, j, w) num_threads(num_tasks)
    {
        int tid, istart, jstart, iend, jend;
        tid = omp_get_thread_num();
        istart = ((tid/task_c) * (N/task_r)%N);
        iend = ((tid/task_c + 1) * (N/task_r)%N);
        if (iend == 0) {iend = N;}
        jstart = ((tid%task_r) * (M/task_c)%M);
        jend = ((tid%task_r + 1) * (M/task_c)%M);
        if (jend == 0) {jend = M;}

        for (i=istart; i<iend; i++) { /* decompose this loop */
            for (j=jstart; j<jend; j++) { /* decompose this loop */
                  REAL temp = 0.0;
                for (w=0; w<K; w++) {
                    temp += A[i*K+w]*B[w*M+j];
                }
                C[i*M+j] = temp;
            }
        }
    } /* end of parallel */
}

/* Parallel For Row */
void mm_parallel_for_row(int N, int K, int M, REAL * A, REAL * B, REAL * C, int num_tasks){
	int i, j, w;
	omp_set_num_threads(num_tasks);
	#pragma omp parallel shared (N, K, M, A, B, C, num_tasks) private (i, j, w)
	{
		#pragma omp for schedule(static) nowait
		for (i=0; i<N; i++) {
			for (j=0; j<M; j++) {
				REAL temp = 0.0;
				for (w=0; w<K; w++)
					temp += A[i*K+w]*B[w*M+j];
				C[i*M+j] = temp;
			}
		}
	} /* end of parallel */
}

/* Parallel For Column */
void mm_parallel_for_col(int N, int K, int M, REAL * A, REAL * B, REAL * C, int num_tasks){
	int i, j, w;
	omp_set_num_threads(num_tasks);
	#pragma omp parallel shared (N, K, M, A, B, C, num_tasks) private (i, j, w)
	{
		for (i=0; i<N; i++) {
			#pragma omp for schedule(static) nowait
			for (j=0; j<M; j++) {
				REAL temp = 0.0;
				for (w=0; w<K; w++)
					temp += A[i*K+w]*B[w*M+j];
				C[i*M+j] = temp;
			}
		}
	} /* end of parallel */
}

/* Parallel For Row Column */
void mm_parallel_for_rowcol(int N, int K, int M, REAL * A, REAL * B, REAL * C, int num_tasks){
	int i, j, w;
	omp_set_num_threads(num_tasks);
	#pragma omp parallel shared (N, K, M, A, B, C, num_tasks) private (i, j, w)
    {
		#pragma omp for collapse(2) schedule(static) nowait
		for (i=0; i<N; i++) {
			for (j=0; j<M; j++) {
				REAL temp = 0.0;
				for (w=0; w<K; w++)
					temp += A[i*K+w]*B[w*M+j];
				C[i*M+j] = temp;
			}
		}
	} /* end of parallel */
}