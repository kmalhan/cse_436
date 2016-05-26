/*
 * Square matrix multiplication
 * A[N][N] * B[N][N] = C[N][N]
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/timeb.h>
#include <string.h>

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

void init(int N, REAL A[][N]) {
    int i, j;

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i][j] = (REAL) drand48();
        }
    }
}

double maxerror(int N, REAL A[][N], REAL B[][N]) {
    int i, j;
    double error = 0.0;

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            double diff = (A[i][j] - B[i][j]) / A[i][j];
            if (diff < 0)
                diff = -diff;
            if (diff > error)
                error = diff;
        }
    }
    return error;
}

void mm(int N, REAL A[][N], REAL B[][N], REAL C[][N]);

int main(int argc, char *argv[]) {
    int N;
    int num_tasks = 4; /* 4 is default number of tasks */
    double elapsed_base;
    if (argc < 2) {
        fprintf(stderr, "Usage: mm <n> [<#tasks(%d)>]\n", num_tasks);
        exit(1);
    }
    N = atoi(argv[1]);
    if (argc > 2) num_tasks = atoi(argv[2]);
    REAL A[N][N];
    REAL B[N][N];
    REAL C_base[N][N];

    srand48((1 << 12));
    init(N, A);
    init(N, B);

    /* example run */
    elapsed_base = read_timer();
    mm(N, A, B, C_base);
    elapsed_base = (read_timer() - elapsed_base);

    printf("======================================================================================================\n");
    printf("\tMatrix Multiplication: A[N][N] * B[N][N] = C[N][N], N=%d\n", N);
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Performance:\t\tRuntime (ms)\t MFLOPS \t\tError (compared to base)\n");
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("mm:\t\t%4f\t%4f \t\t%g\n", elapsed_base * 1.0e3, ((((2.0 * N) * N) * N) / (1.0e6 * elapsed_base)), maxerror(N, C_base, C_base));
}

void mm(int N, REAL A[][N], REAL B[][N], REAL C[][N]) {
int i, j, k;
for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            C[i][j] = 0;
            for (k = 0; k < N; k++)
                C[i][j] += A[i][k]*B[k][j];
        }
    }
}
