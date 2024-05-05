
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


void mult_mat_host(double* A, double* B, double* C, int N)
{
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			for (int k = 0; k < N; k++)
			{
				C[i * N + j] += A[i * N + k] * B[k * N + j];
			}
}

__global__ void mult_mat_device(double* A, double* B, double* C, int N)
{
	unsigned int line_id = threadIdx.x + blockIdx.x * blockDim.x;
	int i = line_id / N;
	int j = line_id % N;
	if (i < N && j < N)
	{
		for (int k = 0; k < N; k++)
		{
			C[i * N + j] += A[i * N + k] * B[k * N + j];
		}
	}
}

int main()
{
	// Объявление переменных
	int N = 1024;
	double* h_A, * h_B, * h_C;
	double* d_A, * d_B, * d_C;
	clock_t t1, t2;
	cudaEvent_t event1, event2;
	float time_device = 0.0f;
	float time_device_copy = 0.0f;
	cudaError_t err;

	// Выделение памяти под рабочие массивы
	h_A = (double*)malloc(N * N * sizeof(double));
	h_B = (double*)malloc(N * N * sizeof(double));
	h_C = (double*)malloc(N * N * sizeof(double));

	cudaMalloc((void**)&d_A, N * N * sizeof(double)); CHECK_ERR("malloc_1\n");
	cudaMalloc((void**)&d_B, N * N * sizeof(double)); CHECK_ERR("malloc_2\n");
	cudaMalloc((void**)&d_C, N * N * sizeof(double)); CHECK_ERR("malloc_3\n");
	cudaEventCreate(&event1);
	cudaEventCreate(&event2);

	// Инициализация начальными данными
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			h_A[i * N + j] = 1.0;
			h_B[i * N + j] = 2.0;
			h_C[i * N + j] = 0.0;
		}

	cudaEventRecord(event1, 0);
	cudaMemcpy(d_A, h_A, N * N * sizeof(double), cudaMemcpyHostToDevice); CHECK_ERR("memcpy_1\n");
	cudaMemcpy(d_B, h_B, N * N * sizeof(double), cudaMemcpyHostToDevice); CHECK_ERR("memcpy_2\n");
	cudaEventRecord(event2, 0);
	cudaEventSynchronize(event2);
	cudaEventElapsedTime(&time_device_copy, event1, event2);
	printf("Time copy from device %le msec\n", time_device_copy);

	// kernel
	int threads = 32;
	int blocks = (N * N) / threads + 1;
	cudaEventRecord(event1, 0);
	mult_mat_device <<<blocks, threads >>> (d_A, d_B, d_C, N); CHECK_ERR("kernel_1\n");		// fine
	//sum_vec_device <<<threads, blocks>>> (d_A, d_B, d_C, N); CHECK_ERR("kernel_2\n");		// error
	cudaEventRecord(event2, 0);
	cudaEventSynchronize(event2); // когда выполниться событие_2, хост разблокируется
	cudaEventElapsedTime(&time_device, event1, event2);
	printf("Time work on device %le msec\n", time_device);

	cudaMemcpy(h_C, d_C, N * N * sizeof(double), cudaMemcpyDeviceToHost); CHECK_ERR("memcpy_3\n");


	// Сложение векторов
	t1 = clock();
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			h_C[i * N + j] = 0.0;
		}
	mult_mat_host(h_A, h_B, h_C, N);
	t2 = clock();
	printf("Time work on host %le msec\n", ((double)(t2 - t1) / CLOCKS_PER_SEC) * 1000);

	// Освобождение памяти
	free(h_A); free(h_B); free(h_C);
	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

//Ctrl + s (сохранение)
//Ctrl + F7
//Ctrl + F5
