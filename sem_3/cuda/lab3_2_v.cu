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

typedef struct
{
	int r;
	int g;
	int b;
} Color;

void fil_AoS_host(Color* h_A, Color* h_B, int N)
{
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			if (i == 0 || i == N - 1 || j == 0 || j == N - 1)
			{
				h_B[i * N + j].r = h_A[i * N + j].r;
				h_B[i * N + j].g = h_A[i * N + j].g;
				h_B[i * N + j].b = h_A[i * N + j].b;
			}
			else
			{
				for (int k = i - 1; k < i + 2; k++)
					for (int z = j - 1; z < j + 2; z++)
					{
						h_B[i * N + j].r += h_A[k * N + z].r;
						h_B[i * N + j].g += h_A[k * N + z].g;
						h_B[i * N + j].b += h_A[k * N + z].b;
					}
				h_B[i * N + j].r = h_B[i * N + j].r / 9;
				h_B[i * N + j].g = h_B[i * N + j].g / 9;
				h_B[i * N + j].b = h_B[i * N + j].b / 9;
			}
}

__global__ void fil_AoS_device(Color* d_A, Color* d_B, int N)
{
	unsigned int line_id = threadIdx.x + blockIdx.x * blockDim.x;
	int i = line_id / N;
	int j = line_id % N;
	if (i < N && j < N)
	{
		if (i == 0 || i == N - 1 || j == 0 || j == N - 1)
		{
			d_B[i * N + j].r = d_A[i * N + j].r;
			d_B[i * N + j].g = d_A[i * N + j].g;
			d_B[i * N + j].b = d_A[i * N + j].b;
		}
		else
		{
			for (int k = i - 1; k < i + 2; k++)
				for (int z = j - 1; z < j + 2; z++)
				{
					d_B[i * N + j].r += d_A[k * N + z].r;
					d_B[i * N + j].g += d_A[k * N + z].g;
					d_B[i * N + j].b += d_A[k * N + z].b;
				}
			d_B[i * N + j].r = d_B[i * N + j].r / 9;
			d_B[i * N + j].g = d_B[i * N + j].g / 9;
			d_B[i * N + j].b = d_B[i * N + j].b / 9;
		}
	}
}

int main()
{
	// Объявление переменных
	int N = 7;
	Color* h_A, * h_B, * d_A, * d_B;
	clock_t t1, t2;
	cudaEvent_t event1, event2;
	float time_device = 0.0f;
	float time_device_copy = 0.0f;
	cudaError_t err;

	// Выделение памяти под рабочие массивы
	h_A = (Color*)malloc(N * N * sizeof(Color));
	h_B = (Color*)malloc(N * N * sizeof(Color));

	cudaMalloc((void**)&d_A, N * N * sizeof(Color)); CHECK_ERR("malloc_1\n");
	cudaMalloc((void**)&d_B, N * N * sizeof(Color)); CHECK_ERR("malloc_1\n");

	cudaEventCreate(&event1);
	cudaEventCreate(&event2);

	// Инициализация начальными данными
	srand(time(0));
	for (int i = 0; i < N * N; i++)
	{
		h_A[i].r = rand() % 256;
		h_A[i].g = rand() % 256;
		h_A[i].b = rand() % 256;
		h_B[i].r = 0;
		h_B[i].g = 0;
		h_B[i].b = 0;
	}
	printf("A host\n");
	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++) {
			printf("%d\t", h_A[i * N + j].r);
		}
		printf("\n");
	}
		/*
		printf("\n");
		printf("h_A_g\n");
		for (int j = 0; j < N; j++) {
			for (int i = 0; i < N; i++) {
				printf("%d\t", h_A[i * N + j].g);
			}
			printf("\n");
		}
		printf("\n");
		printf("h_A_b\n");
		for (int j = 0; j < N; j++) {
			for (int i = 0; i < N; i++) {
				printf("%d\t", h_A[i * N + j].b);
			}
			printf("\n");
		}
	*/
	cudaEventRecord(event1, 0);
	cudaMemcpy(d_A, h_A, N * N * sizeof(Color), cudaMemcpyHostToDevice); CHECK_ERR("memcpy_1\n");
	cudaEventRecord(event2, 0);
	cudaEventSynchronize(event2);
	cudaEventElapsedTime(&time_device_copy, event1, event2);
	printf("Time copy from device %le msec\n", time_device_copy);

	// kernel
	int threads = 32;
	int blocks = (N * N) / threads + 1;
	cudaEventRecord(event1, 0);
	fil_AoS_device << <blocks, threads >> > (d_A, d_B, N); CHECK_ERR("kernel_1\n");		// fine
	cudaEventRecord(event2, 0);
	cudaEventSynchronize(event2); // когда выполниться событие_2, хост разблокируется
	cudaEventElapsedTime(&time_device, event1, event2);
	printf("Time work on device %le msec\n", time_device);

	cudaMemcpy(h_B, d_B, N * N * sizeof(Color), cudaMemcpyDeviceToHost); CHECK_ERR("memcpy_2\n");
	printf("B device\n");
	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++) {
			printf("%d\t", h_B[i * N + j].r);
		}
		printf("\n");
	}
		/*
		printf("\n");
		printf("d_B_g\n");
		for (int j = 0; j < N; j++) {
			for (int i = 0; i < N; i++) {
				printf("%d\t", h_B[i * N + j].g);
			}
			printf("\n");
		}
		printf("\n");
		printf("d_B_b\n");
		for (int j = 0; j < N; j++) {
			for (int i = 0; i < N; i++) {
				printf("%d\t", h_B[i * N + j].b);
			}
			printf("\n");
		}
	*/
	// Фильтрация
	t1 = clock();
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			h_B[i * N + j].r = 0.0;
			h_B[i * N + j].g = 0.0;
			h_B[i * N + j].b = 0.0;
		}
	fil_AoS_host(h_A, h_B, N);
	t2 = clock();
	printf("Time work on host %le msec\n", ((double)(t2 - t1) / CLOCKS_PER_SEC) * 1000);
	printf("B host\n");
	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++) {
			printf("%d\t", h_B[i * N + j].r);
		}
		printf("\n");
	}
		/*
		printf("\n");
		printf("h_B_g\n");
		for (int j = 0; j < N; j++) {
			for (int i = 0; i < N; i++) {
				printf("%d\t", h_B[i * N + j].g);
			}
			printf("\n");
		}
		printf("\n");
		printf("h_B_b\n");
		for (int j = 0; j < N; j++) {
			for (int i = 0; i < N; i++) {
				printf("%d\t", h_B[i * N + j].b);
			}
			printf("\n");
		}
	*/
	// Освобождение памяти
	free(h_A); free(h_B);
	cudaFree(d_A); cudaFree(d_B);
}
