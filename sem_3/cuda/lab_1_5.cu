
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifdef _DEBUG

#define CHECK_ERR(a) { err = cudaGetLastError(); \
				 if(err != cudaSuccess) { printf(a); printf("%s(%d): %s \n", __FILE__, __LINE__, cudaGetErrorString(err)); } }
#else
#define CHECK_ERR(a)
#endif


void sum_vec_host(double* A, double* B, double* C, int N)
{
	for (int i = 0; i < N; i++)
	{
		C[i] = A[i] + B[i];
	}
}

__global__ void sum_vec_device(double* A, double* B, double* C, int N)
{
	unsigned int line_id = threadIdx.x + blockIdx.x * blockDim.x;
	if (line_id < N)
	{
		C[line_id] = A[line_id] + B[line_id];
	}
}

int main()
{
	// Объявление переменных
	int N = 1000000;
	double* h_A, * h_B, * h_C;
	double* d_A, * d_B, * d_C;
	clock_t t1, t2;
	cudaEvent_t event1, event2;
	float time_device = 0.0f;
	float time_device_copy = 0.0f;
	cudaError_t err;

	cudaStream_t stream[2];
	cudaStreamCreate(&stream[0]);
	cudaStreamCreate(&stream[1]);

	// Выделение памяти под рабочие массивы
	cudaMallocHost((void**)&h_A, N * sizeof(double)); CHECK_ERR("malloc_1\n");
	cudaMallocHost((void**)&h_B, N * sizeof(double)); CHECK_ERR("malloc_1\n");
	cudaMallocHost((void**)&h_C, N * sizeof(double)); CHECK_ERR("malloc_1\n");

	cudaMalloc((void**)&d_A, N * sizeof(double)); CHECK_ERR("malloc_1\n");
	cudaMalloc((void**)&d_B, N * sizeof(double)); CHECK_ERR("malloc_2\n");
	cudaMalloc((void**)&d_C, N * sizeof(double)); CHECK_ERR("malloc_3\n");
	cudaEventCreate(&event1);
	cudaEventCreate(&event2);

	// Инициализация начальными данными
	for (int i = 0; i < N; i++)
	{
		h_A[i] = 1.0;
		h_B[i] = 2.0;
		h_C[i] = 0.0;
	}

	cudaEventRecord(event1, 0);
	for (int i = 0; i < 2; i++)
	{
		cudaMemcpyAsync(d_A + i * N / 2, h_A + i * N / 2, (N / 2) * sizeof(double), cudaMemcpyHostToDevice, stream[i]); CHECK_ERR("memcpy_1\n");
		cudaMemcpyAsync(d_B + i * N / 2, h_B + i * N / 2, (N / 2) * sizeof(double), cudaMemcpyHostToDevice, stream[i]); CHECK_ERR("memcpy_2\n");
	}

	// kernel
	int threads = 32;
	int blocks = N / (2 * threads) + 1;
	for (int i = 0; i < 2; i++)
	{
		sum_vec_device <<<blocks, threads, 0, stream[i]>>> (d_A + i * N / 2, d_B + i * N / 2, d_C + i * N / 2, N / 2); CHECK_ERR("kernel_1\n");
	}

	for (int i = 0; i < 2; i++)
	{
		cudaMemcpyAsync(d_C + i * N / 2, h_C + i * N / 2, (N / 2) * sizeof(double), cudaMemcpyHostToDevice, stream[i]); CHECK_ERR("memcpy_3\n");
	}

	cudaEventRecord(event2, 0);
	cudaEventSynchronize(event2);
	cudaEventElapsedTime(&time_device_copy, event1, event2);
	printf("Time work on device %le msec\n", time_device_copy);
	
	// Сложение векторов
	t1 = clock();
	sum_vec_host(h_A, h_B, h_C, N);
	t2 = clock();
	printf("Time work on host %le msec\n", ((double)(t2 - t1) / CLOCKS_PER_SEC) * 1000);

	// Освобождение памяти
	cudaFree(h_A); cudaFree(h_B); cudaFree(h_C);
	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
	for (int i = 0; i < 2; i++)
	{
		cudaStreamDestroy(stream[i]);
	}
}

//Ctrl + s (сохранение)
//Ctrl + F7
//Ctrl + F5
