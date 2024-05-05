#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

#define L 5					// толщина пластины
#define sigma 1				// константа


#ifdef _DEBUG

#define CHECK_ERR(a) { err = cudaGetLastError(); \
				 if(err != cudaSuccess) { printf(a); printf("%s(%d): %s \n", __FILE__, __LINE__, cudaGetErrorString(err)); } }
#else
#define CHECK_ERR(a)
#endif


void host(int N)
{
	double x, l, csi;
	int t = 0;

	while (N > 0)
	{
		x = 0;
		csi = (double)rand() / (double)(RAND_MAX + 1);
		l = -log(csi) / sigma;
		x += l;
		if (x > L)
		{
			t++;
		}
		N--;
	}
	printf("%d\n", t);
}


__global__ void device(int N, int* Nplus)
{
	unsigned int seed = threadIdx.x + blockIdx.x * blockDim.x;
	curandState s;
	curand_init(seed, 0, 0, &s);
	float csi = curand_uniform(&s);
	double x, l;
	int t = 0;

	x = 0;
	l = -log(csi) / sigma;
	x += l;
	if (x > L)
	{
		atomicAdd(Nplus, 1);
	}
}



int main()
{
	int N = 1000000;				// число испытаний
	srand(time(NULL));
	clock_t t1, t2;
	cudaEvent_t event1, event2;
	float time_device = 0.0f;
	float time_device_copy = 0.0f;
	cudaError_t err;

	t1 = clock();
	host(N);
	t2 = clock();
	printf("Time work on host %lf msec\n", (double)(t2 - t1) / CLOCKS_PER_SEC * 1000);

	cudaEventCreate(&event1);
	cudaEventCreate(&event2);

	int* Nplus;
	cudaMalloc(&Nplus, sizeof(int));

	int threads = 32;
	int blocks = N / threads + 1;

	cudaEventRecord(event1, 0);
	device << <blocks, threads >> > (N, Nplus); CHECK_ERR("kernel_1\n");
	cudaEventRecord(event2, 0);
	cudaEventSynchronize(event2);
	cudaEventElapsedTime(&time_device, event1, event2);

	cudaMemcpy(&Nplus, Nplus, sizeof(int), cudaMemcpyDeviceToHost);
	printf("%d\n", Nplus);
	printf("Time work on device %lf msec\n", time_device); CHECK_ERR("kernel_15\n");

	cudaFree(Nplus);

	int N_a = exp(-L) * N;
	printf("%d\n", N_a);

	return 0;
}
