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


// Формула рассчета значений в матрице B:
void fil_SoA_host(double* h_A_r, double* h_A_g, double* h_A_b, double* h_B_r, double* h_B_g, double* h_B_b, int N)
{
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			if (i == 0 || i == N - 1 || j == 0 || j == N - 1)
			{
				h_B_r[i * N + j] = h_A_r[i * N + j];
				h_B_g[i * N + j] = h_A_g[i * N + j];
				h_B_b[i * N + j] = h_A_b[i * N + j];
			}
			else
			{
				for (int k = i - 1; k < i + 2; k++)
					for (int z = j - 1; z < j + 2; z++)
					{
						h_B_r[i * N + j] += h_A_r[k * N + z];
						h_B_g[i * N + j] += h_A_g[k * N + z];
						h_B_b[i * N + j] += h_A_b[k * N + z];
					}
				h_B_r[i * N + j] = h_B_r[i * N + j] / 9;
				h_B_g[i * N + j] = h_B_g[i * N + j] / 9;
				h_B_b[i * N + j] = h_B_b[i * N + j] / 9;
			}
}

__global__ void fil_SoA_device(double* d_A_r, double* d_A_g, double* d_A_b, double* d_B_r, double* d_B_g, double* d_B_b, int N)
{
	unsigned int line_id = threadIdx.x + blockIdx.x * blockDim.x;
	int i = line_id / N;
	int j = line_id % N;
	if (i < N && j < N)
	{
		if (i == 0 || i == N - 1 || j == 0 || j == N - 1)
		{
			d_B_r[i * N + j] = d_A_r[i * N + j];
			d_B_g[i * N + j] = d_A_g[i * N + j];
			d_B_b[i * N + j] = d_A_b[i * N + j];
		}
		else
		{
			for (int k = i - 1; k < i + 2; k++)
				for (int z = j - 1; z < j + 2; z++)
				{
					d_B_r[i * N + j] += d_A_r[k * N + z];
					d_B_g[i * N + j] += d_A_g[k * N + z];
					d_B_b[i * N + j] += d_A_b[k * N + z];
				}
			d_B_r[i * N + j] = d_B_r[i * N + j] / 9;
			d_B_g[i * N + j] = d_B_g[i * N + j] / 9;
			d_B_b[i * N + j] = d_B_b[i * N + j] / 9;
		}
	}
}

int main()
{
	// Объявление переменных
	int N = 1024;
	double* h_A_r, * h_A_g, * h_A_b, * h_B_r, * h_B_g, * h_B_b;
	double* d_A_r, * d_A_g, * d_A_b, * d_B_r, * d_B_g, * d_B_b;
	clock_t t1, t2;
	cudaEvent_t event1, event2;
	float time_device = 0.0f;
	float time_device_copy = 0.0f;
	cudaError_t err;

	// Выделение памяти под рабочие массивы
	h_A_r = (double*)malloc(N * N * sizeof(double));
	h_A_g = (double*)malloc(N * N * sizeof(double));
	h_A_b = (double*)malloc(N * N * sizeof(double));
	h_B_r = (double*)malloc(N * N * sizeof(double));
	h_B_g = (double*)malloc(N * N * sizeof(double));
	h_B_b = (double*)malloc(N * N * sizeof(double));

	cudaMalloc((void**)&d_A_r, N * N * sizeof(double)); CHECK_ERR("malloc_1\n");
	cudaMalloc((void**)&d_A_g, N * N * sizeof(double)); CHECK_ERR("malloc_1\n");
	cudaMalloc((void**)&d_A_b, N * N * sizeof(double)); CHECK_ERR("malloc_1\n");
	cudaMalloc((void**)&d_B_r, N * N * sizeof(double)); CHECK_ERR("malloc_1\n");
	cudaMalloc((void**)&d_B_g, N * N * sizeof(double)); CHECK_ERR("malloc_1\n");
	cudaMalloc((void**)&d_B_b, N * N * sizeof(double)); CHECK_ERR("malloc_1\n");
	cudaEventCreate(&event1);
	cudaEventCreate(&event2);

	// Инициализация начальными данными
	srand(time(0));
	for (int i = 0; i < N * N; i++)
	{
		h_A_r[i] = rand() % 256;
		h_A_g[i] = rand() % 256;
		h_A_b[i] = rand() % 256;
		h_B_r[i] = 0;
		h_B_g[i] = 0;
		h_B_b[i] = 0;
	}
/*
		printf("h_A_r\n");
		for (int j = 0; j < N; j++) {
			for (int i = 0; i < N; i++) {
				printf("%lf\t", h_A_r[i * N + j]);
			}
			printf("\n");
		}
		printf("\n");
		printf("h_A_g\n");
		for (int j = 0; j < N; j++) {
			for (int i = 0; i < N; i++) {
				printf("%lf\t", h_A_g[i * N + j]);
			}
			printf("\n");
		}
		printf("\n");
		printf("h_A_b\n");
		for (int j = 0; j < N; j++) {
			for (int i = 0; i < N; i++) {
				printf("%lf\t", h_A_b[i * N + j]);
			}
			printf("\n");
		}
*/	

	cudaEventRecord(event1, 0);
	cudaMemcpy(d_A_r, h_A_r, N * N * sizeof(double), cudaMemcpyHostToDevice); CHECK_ERR("memcpy_1\n");
	cudaMemcpy(d_A_g, h_A_g, N * N * sizeof(double), cudaMemcpyHostToDevice); CHECK_ERR("memcpy_2\n");
	cudaMemcpy(d_A_b, h_A_b, N * N * sizeof(double), cudaMemcpyHostToDevice); CHECK_ERR("memcpy_3\n");
	cudaEventRecord(event2, 0);
	cudaEventSynchronize(event2);
	cudaEventElapsedTime(&time_device_copy, event1, event2);
	printf("Time copy to device %le msec\n", time_device_copy);

	// kernel
	int threads = 32;
	int blocks = (N * N) / threads + 1;
	cudaEventRecord(event1, 0);
	fil_SoA_device <<<blocks, threads >> > (d_A_r, d_A_g, d_A_b, d_B_r, d_B_g, d_B_b, N); CHECK_ERR("kernel_1\n");		// fine
	cudaEventRecord(event2, 0); CHECK_ERR("kernel_12\n");
	cudaEventSynchronize(event2); CHECK_ERR("kernel_13\n"); // когда выполниться событие_2, хост разблокируется
	cudaEventElapsedTime(&time_device, event1, event2); CHECK_ERR("kernel_14\n");
	printf("Time work on device %le msec\n", time_device); CHECK_ERR("kernel_15\n");

	cudaMemcpy(h_B_r, d_B_r, N * N * sizeof(double), cudaMemcpyDeviceToHost); CHECK_ERR("memcpy_4\n");
	cudaMemcpy(h_B_g, d_B_g, N * N * sizeof(double), cudaMemcpyDeviceToHost); CHECK_ERR("memcpy_5\n");
	cudaMemcpy(h_B_b, d_B_b, N * N * sizeof(double), cudaMemcpyDeviceToHost); CHECK_ERR("memcpy_6\n");

	/*
		printf("d_B_r\n");
		for (int j = 0; j < N; j++) {
			for (int i = 0; i < N; i++) {
				printf("%lf\t", h_B_r[i * N + j]);
			}
			printf("\n");
		}
		printf("\n");
		printf("d_B_g\n");
		for (int j = 0; j < N; j++) {
			for (int i = 0; i < N; i++) {
				printf("%lf\t", h_B_g[i * N + j]);
			}
			printf("\n");
		}
		printf("\n");
		printf("d_B_b\n");
		for (int j = 0; j < N; j++) {
			for (int i = 0; i < N; i++) {
				printf("%lf\t", h_B_b[i * N + j]);
			}
			printf("\n");
		}
	*/
	// Фильтрация
	t1 = clock();
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			h_B_r[i * N + j] = 0.0;
			h_B_g[i * N + j] = 0.0;
			h_B_b[i * N + j] = 0.0;
		}
	fil_SoA_host(h_A_r, h_A_g, h_A_b, h_B_r, h_B_g, h_B_b, N);
	t2 = clock();
	printf("Time work on host %le msec\n", ((double)(t2 - t1) / CLOCKS_PER_SEC) * 1000);
/*
		printf("h_B_r\n");
		for (int j = 0; j < N; j++) {
			for (int i = 0; i < N; i++) {
				printf("%lf\t", h_B_r[i * N + j]);
			}
			printf("\n");
		}
		printf("\n");
		printf("h_B_g\n");
		for (int j = 0; j < N; j++) {
			for (int i = 0; i < N; i++) {
				printf("%lf\t", h_B_g[i * N + j]);
			}
			printf("\n");
		}
		printf("\n");
		printf("h_B_b\n");
		for (int j = 0; j < N; j++) {
			for (int i = 0; i < N; i++) {
				printf("%lf\t", h_B_b[i * N + j]);
			}
			printf("\n");
		}
*/
	// Освобождение памяти
	free(h_A_r); free(h_A_g); free(h_A_b); free(h_B_r); free(h_B_g); free(h_B_b);
	cudaFree(d_A_r); cudaFree(d_A_g); cudaFree(d_A_b); cudaFree(d_B_r); cudaFree(d_B_g); cudaFree(d_B_b);
}
