#include <stdio.h>

__global__ void hellokernel()
{
	printf("Hello World!\n");
}

int main(void)
{
	int num_threads = 10;
	int num_blocks = 10;
	hellokernel<<<num_blocks,num_threads>>>();
	cudaDeviceSynchronize();

	return 0;
}
