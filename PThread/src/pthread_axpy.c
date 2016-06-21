/*
 * AXPY  Y[N] = Y[N] + a*X[N]
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>

#include <pthread.h>

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
void init(REAL A[], int N) {
    int i;
    for (i = 0; i < N; i++) {
        A[i] = (double) drand48();
    }
}

/* for checking whether the results are correct or not */
double check(REAL A[], REAL B[], int N) {
    int i;
    double sum = 0.0;
    for (i = 0; i < N; i++) {
        sum += A[i] - B[i];
    }
    return sum;
}

/* function pre-declaration, implementation are after the main function */
void axpy_base(int N, REAL Y[], REAL X[], REAL a);
void axpy_base_sub(int i_start, int Nt, int N, REAL Y[], REAL X[], REAL a);
void axpy_dist(int N, REAL Y[], REAL X[], REAL a, int num_tasks);
void axpy_pthread(int N, REAL Y[], REAL X[], REAL a, int num_tasks);

int main(int argc, char *argv[]) {
    int N = VECTOR_LENGTH;
    int num_tasks = 4; /* 4 is default number of tasks */
    double elapsed; /* for timing */
    double elapsed_pthread; /* for timing */
    if (argc < 2) {
        fprintf(stderr, "Usage: axpy <n> [<#tasks(%d)>]\n", num_tasks);
        exit(1);
    }
    N = atoi(argv[1]); /* read in the first argument as the size of array */
    if (argc > 2) num_tasks = atoi(argv[2]); /* the second optional argu as the num_tasks */
    REAL a = 123.456;
    REAL Y_base[N];
    REAL Y_pthread[N];
    REAL X[N];

    /* init the array */
    srand48((1 << 12));
    init(X, N);
    init(Y_base, N);
    memcpy(Y_pthread, Y_base, N * sizeof(REAL));

    /* example run */
    elapsed = read_timer();
    axpy_base(N, Y_base, X, a);
    elapsed = (read_timer() - elapsed);

    elapsed_pthread = read_timer();
    axpy_pthread(N, Y_pthread, X, a, num_tasks);
    elapsed_pthread = (read_timer() - elapsed_pthread);
    
    /* you should add the call to each function and time the execution */
    printf("======================================================================================================\n");
    printf("\tAXPY: Y[N] = Y[N] + a*X[N], N=%d, %d threads/tasks for dist\n", N, num_tasks);
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Performance:\t\tRuntime (ms)\t MFLOPS \t\tError (compared to base)\n");
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("axpy_base:\t\t%4f\t%4f \t\t%g\n", elapsed * 1.0e3, (2.0 * N) / (1.0e6 * elapsed), check(Y_base, Y_base, N));
    printf("axpy_pthread:\t\t%4f\t%4f \t\t%g\n", elapsed_pthread * 1.0e3, (2.0 * N) / (1.0e6 * elapsed_pthread), check(Y_base, Y_pthread, N));
    return 0;
}

/* the serial version of axpy */
void axpy_base(int N, REAL Y[], REAL X[], REAL a) {
    int i;
    for (i = 0; i < N; ++i)
        Y[i] += a * X[i];
}

/* the serial verson of axpy to compute subarray with i_start position for Nt elements */
void axpy_base_sub(int i_start, int Nt, int N, REAL Y[], REAL X[], REAL a) {
    int i;
    for (i = i_start; i < i_start + Nt; ++i)
        Y[i] += a * X[i];
}

/* this function block decomposing N among num_tasks for tid
 * it will return Nt and start, which are the sizes of the subarray and the 
 * start position of the subarray
 *
 * The algorithm works also if N is NOT divisible by num_tasks by giving the first
 * N%num_tasks tasks each one more element. 
 */
void dist(int tid, int N, int num_tasks, int *Nt, int *start) {
    int remain = N % num_tasks;
    int esize = N / num_tasks;
    if (tid < remain) { /* each of the first remain task has one more element */
        *Nt = esize + 1;
        *start = *Nt * tid;
    } else {
        *Nt = esize;
        *start = esize * tid + remain;
    }
}

/* the serial version demonstrating the way of decomposing the axpy into multiple tasks */
void axpy_dist(int N, REAL Y[], REAL X[], REAL a, int num_tasks) {
    int tid;
    for (tid = 0; tid < num_tasks; tid++) {
        int Nt, start;
        dist(tid, N, num_tasks, &Nt, &start);
        axpy_base_sub(start, Nt, N, Y, X, a);
    }
}

/* thread argument data structure */
struct axpy_pthread_data {
    int Nt; /* the size of subarray to compute */
    int start; /* the start position of the subarray */
    int N; /* the size of the full array */
    REAL *Y;
    REAL *X;
    REAL a;
};

/* the pthread function */
void * axpy_thread_func(void * axpy_thread_arg) {
    struct axpy_pthread_data * arg = (struct axpy_pthread_data *) axpy_thread_arg;
    axpy_base_sub(arg->start, arg->Nt, arg->N, arg->Y, arg->X, arg->a);
    pthread_exit(NULL);
}

/* this function performs distribution of N onto num_tasks tasks and create the same
 * amount of pthreads, each to compute one task */
void axpy_pthread(int N, REAL Y[], REAL X[], REAL a, int num_tasks) {
    struct axpy_pthread_data pthread_data_array[num_tasks];
    pthread_t task_threads[num_tasks];
    int tid;
    for (tid = 0; tid < num_tasks; tid++) {
        int Nt, start;
	/* decompositio to get portion of array for computation */
        dist(tid, N, num_tasks, &Nt, &start);

	/* init pthread function arguments */
        struct axpy_pthread_data *task_data = &pthread_data_array[tid];
        task_data->start = start;
        task_data->Nt = Nt;
        task_data->a = a;
        task_data->X = X;
        task_data->Y = Y;
        task_data->N = N;

	/* create pthread */
        pthread_create(&task_threads[tid], NULL, axpy_thread_func, (void*)task_data);
    }

    /* join the pthread */
    for (tid = 0; tid < num_tasks; tid++) {
        pthread_join(task_threads[tid], NULL);
    }
}
