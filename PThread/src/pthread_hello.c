#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#define NUM_THREADS	5

void *PrintHello(void *thread_id) {
    long tid = (long)thread_id;
    printf("Hello World! It's me, thread #%ld!\n", tid);
    pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
    pthread_t threads[NUM_THREADS];
    long t;
    for(t=0;t<NUM_THREADS;t++) {
       printf("In main: creating thread %ld\n", t);
       int rc = pthread_create(&threads[t], NULL, PrintHello, (void *)t );
       if (rc) {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
       }
    }
  pthread_exit(NULL);
}

