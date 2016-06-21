#include <pthread.h>
#define NUM_THREADS	8

struct thread_data {
   int	 thread_id;
   char *message;
};

struct thread_data thread_data_array[NUM_THREADS];

void *PrintHello(void *threadarg) {
   int taskid;
   char *hello_msg;

   sleep(1);
   struct thread_data *my_data = (struct thread_data *) threadarg;
   taskid = my_data->thread_id;
   hello_msg = my_data->message;
   printf("Thread %d: %s\n", taskid, hello_msg);
   pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
    pthread_t threads[NUM_THREADS];
    int t;
    char *messages[NUM_THREADS];
    messages[0] = "English: Hello World!";
    messages[1] = "French: Bonjour, le monde!";
    messages[2] = "Spanish: Hola al mundo";
    messages[3] = "Klingon: Nuq neH!";
    messages[4] = "German: Guten Tag, Welt!"; 
    messages[5] = "Russian: Zdravstvytye, mir!";
    messages[6] = "Japan: Sekai e konnichiwa!";
    messages[7] = "Latin: Orbis, te saluto!";

    for(t=0;t<NUM_THREADS;t++) {
       struct thread_data * thread_arg = &thread_data_array[t];
       thread_arg->thread_id = t;
       thread_arg->message = messages[t];
	pthread_create(&threads[t], NULL, PrintHello, (void *) thread_arg);
    }
    pthread_exit(NULL);
}

