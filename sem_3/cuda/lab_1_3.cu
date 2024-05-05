
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

	// Выделение памяти под рабочие массивы
	h_A = (double*)malloc(N * sizeof(double));
	h_B = (double*)malloc(N * sizeof(double));
	h_C = (double*)malloc(N * sizeof(double));

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
		//if (i < 10) printf(" %le %le %le\n", h_A[i], h_B[i], h_C[i]);
	}

	cudaEventRecord(event1, 0);
	cudaMemcpy(d_A, h_A, N * sizeof(double), cudaMemcpyHostToDevice); CHECK_ERR("memcpy_1\n");
	cudaMemcpy(d_B, h_B, N * sizeof(double), cudaMemcpyHostToDevice); CHECK_ERR("memcpy_2\n");
	cudaEventRecord(event2, 0);
	cudaEventSynchronize(event2);
	cudaEventElapsedTime(&time_device_copy, event1, event2);
	printf("Time copy from device %le msec\n", time_device_copy);

	// kernel
	int threads = 32;
	int blocks = N / threads + 1;
	cudaEventRecord(event1, 0);
	sum_vec_device << <blocks, threads >> > (d_A, d_B, d_C, N); CHECK_ERR("kernel_1\n");		// fine
	//sum_vec_device <<<threads, blocks>>> (d_A, d_B, d_C, N); CHECK_ERR("kernel_2\n");		// error
	cudaEventRecord(event2, 0);
	cudaEventSynchronize(event2); // когда выполниться событие_2, хост разблокируется
	cudaEventElapsedTime(&time_device, event1, event2);
	printf("Time work on device %le msec\n", time_device);

	cudaMemcpy(d_C, h_C, N * sizeof(double), cudaMemcpyHostToDevice); CHECK_ERR("memcpy_3\n");


	// Сложение векторов
	t1 = clock();
	sum_vec_host(h_A, h_B, h_C, N);
	t2 = clock();
	printf("Time work on host %le msec\n", ((double)(t2 - t1) / CLOCKS_PER_SEC) * 1000);

	// Освобождение памяти
	free(h_A); free(h_B); free(h_C);
	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

//Ctrl + s (сохранение)
//Ctrl + F7
//Ctrl + F5
