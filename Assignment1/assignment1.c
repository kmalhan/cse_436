/********************************
 *
 *  Assignment 1 - CSE436
 *  Kazumi Malhan
 *  05/15/2016
 *
 ********************************/

// Include files
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>

// Macro
#define REAL float
#define VECTOR_LENGTH 512

// Function Prototype
/* C[N][M] = A[N][M] + B[N][M] */
void matrix_addition(int N, int M, REAL* A, REAL* B, REAL* C, int A_rowMajor, int B_rowMajor);
/* C[N][M] = A[N][K] * B[K][M] */
void matrix_multiplication(int N, int K, int M, REAL* A, REAL* B, REAL* C, int A_rowMajor, int B_rowMajor);
/* C[N] = A[N][M] * B[M] */
void mv_multiplication (int N, int M, REAL* A, REAL* B, REAL* C, int A_rowMajor);

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

/* initialize a vector with random floating point numbers */
void init(REAL A[], int N) {
    int i;
    for (i = 0; i < N; i++) {
        A[i] = (double) drand48();
    }
}

/**
 * To compile: gcc assignment1.c -o assignment1
 * To run: ./assignment1 256 128
 *
 */
int main(int argc, char *argv[]) {
    // Local variables
    int N = VECTOR_LENGTH;
    int M = M;
    double elapsed; /* for timing */
    if (argc < 3) {
        fprintf(stderr, "Usage: assignment1 <n> <m> (default %d)\n", N);
        exit(1);
    }
    // Read command line input
    N = atoi(argv[1]);
    M = atoi(argv[2]);
    // Allocate arrays dynamically
    REAL* A = malloc(sizeof(REAL) * N * M);
    REAL* B = malloc(sizeof(REAL) * N * M);
    REAL* C = malloc(sizeof(REAL) * N * M);
    // Initialize arrays
    srand48((1 << 12));
    init(A, N * M);
    init(B, N * M);

    // Run matrix_addition with four variants
    double elapsed_matrixAdd_row_row = read_timer();
    matrix_addition(N, M, A, B, C, 1, 1);
    elapsed_matrixAdd_row_row  = (read_timer() - elapsed_matrixAdd_row_row);

    double elapsed_matrixAdd_row_col = read_timer();
    matrix_addition(N, M, A, B, C, 1, 0);
    elapsed_matrixAdd_row_col  = (read_timer() - elapsed_matrixAdd_row_col);

    double elapsed_matrixAdd_col_row = read_timer();
    matrix_addition(N, M, A, B, C, 0, 1);
    elapsed_matrixAdd_col_row = (read_timer() - elapsed_matrixAdd_col_row);

    double elapsed_matrixAdd_col_col = read_timer();
    matrix_addition(N, M, A, B, C, 0, 0);
    elapsed_matrixAdd_col_col = (read_timer() - elapsed_matrixAdd_col_col);

    // Free allocated memory
    free(A);
    free(B);
    free(C);

    /* Print out results */
    printf("======================================================================================================\n");
    printf("\tN: %d, M: %d, K: %d\n", N, M, N);
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Performance:\t\t\t\tRuntime (ms)\t MFLOPS \n");
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("matrix addition row row:\t\t%4f\t%4f\n",  elapsed_matrixAdd_row_row * 1.0e3,
		    M*N / (1.0e6 *  elapsed_matrixAdd_row_row));
    printf("matrix addition row col:\t\t%4f\t%4f\n",  elapsed_matrixAdd_row_col * 1.0e3,
		    M*N / (1.0e6 *  elapsed_matrixAdd_row_col));
    printf("matrix addition col row:\t\t%4f\t%4f\n",  elapsed_matrixAdd_col_row * 1.0e3,
		    M*N / (1.0e6 *  elapsed_matrixAdd_col_row));
    printf("matrix addition col col:\t\t%4f\t%4f\n",  elapsed_matrixAdd_col_col * 1.0e3,
		    M*N / (1.0e6 *  elapsed_matrixAdd_col_col));
    return 0;
} // end of function

void matrix_addition(int N, int M, REAL* A, REAL* B, REAL* C, int A_rowMajor, int B_rowMajor) {
	if (A_rowMajor != 0 && B_rowMajor != 0) { /* A is row major, B is row major */
		int i, j;
		for (i=0; i<N; i++) {
			for (j=0; j<M; j++) {
				/* the offset of matrix A[i][j] in memory based on A */
			  int offset = i * M + j;
		    C[offset] = A[offset] + B[offset];
			}
    }
	} else if (A_rowMajor != 0 && B_rowMajor == 0) { /* A is row major and B is col major */
		int i, j;
		for (i=0; i<N; i++) {
			for (j=0; j<M; j++) {
				/* the offset of matrix A[i][j] in memory based on A */
			  int rowMajor_offset = i * M + j;
			  int colMajor_offset = j * N + i;
		    C[rowMajor_offset] = A[rowMajor_offset] + B[colMajor_offset];
			}
    }
	} else if (A_rowMajor == 0 && B_rowMajor != 0) { /* A is col major and B is row major */
    int i, j;
    for (i=0; i<N; i++) {
      for (j=0; j<M; j++) {
        /* the offset of matrix A[i][j] in memroy based on A */
        int rowMajor_offset = i * M + j;
        int colMajor_offset = j * N + i;
        C[rowMajor_offset] = A[colMajor_offset] + B[rowMajor_offset];
      }
    }
  } else { /* A is col major and B is col major */
    int i, j;
    for (i=0; i<N; i++) {
      for (j=0; j<M; j++) {
        /* the offset of matrix A[i][j] in memory based on A */
        int rowMajor_offset = i * M + j;
        int colMajor_offset = j * N + i;
        C[rowMajor_offset] = A[colMajor_offset] + B[colMajor_offset];
      }
    }
  }
} // end of function

void matrix_multiplication(int N, int K, int M, REAL* A, REAL* B, REAL* C, int A_rowMajor, int B_rowMajor) {
  if (A_rowMajor != 0 && B_rowMajor != 0) {

  } else if (A_rowMajor != 0 && B_rowMajor == 0) {

  } else if (A_rowMajor == 0 && B_rowMajor != 0) {

  } else {

  }
} // end of function

void mv_multiplication (int N, int M, REAL* A, REAL* B, REAL* C, int A_rowMajor) {
  if (A_rowMajor != 0) {

  } else {

  }
} // end of function
