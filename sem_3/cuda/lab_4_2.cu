#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 11							// размер сетки
#define M 21 
#define hx 1.0 / (M - 1)
#define hy 1.0 / (N	- 1)				// шаг по сетке
//#define T 0.5							// время до которого считаем
#define a 1.0							// константа
#define tau (1 / (3 * a)) * (hx * hx * hy * hy)	/ (hx * hx + hy * hy)	// шаг по времени
#define T tau * 5
#define Q(t, x, y) 0.0					// константа
#define h_A(i, j) h_A[(i) * M + (j)]
#define h_B(i, j) h_B[(i) * M + (j)]
#define d_A(i, j) d_A[(i) * M + (j)]
#define d_B(i, j) d_B[(i) * M + (j)]

#ifdef _DEBUG

#define CHECK_ERR(a) { err = cudaGetLastError(); \
				 if(err != cudaSuccess) { printf(a); printf("%s(%d): %s \n", __FILE__, __LINE__, cudaGetErrorString(err)); } }
#else
#define CHECK_ERR(a)
#endif

void print_matrix(double* h_A, double t)
{
	printf("t = %lf \n", t);
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < M; j++)
		{
			printf("%.3lf ", h_A(i, j));
		}
		printf("\n");
	}
	printf("\n");
}

void U_0(double* h_A)
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < M; j++)
		{
			if(i == N/2 && j == M/2)
				h_A(i, j) = 1;
			else
				h_A(i, j) = 0;
		}
	}
}

void U_n_host(double* h_A, double* h_B, double t)
{
	for (int i = 1; i < N - 1; i++)
	{
		for (int j = 1; j < M - 1; j++)
		{
			double x = j * hx;
			double y = i * hy;
			h_B(i, j) = h_A(i, j) + tau * (a * a * ((h_A(i, j + 1) - 2 * h_A(i, j) + h_A(i, j - 1)) / (hx * hx) + (h_A(i + 1, j) - 2 * h_A(i, j) + h_A(i - 1, j)) / (hy * hy)) + Q(t, x, y));
		}
	}
}


__global__ void U_n_device(double* d_A, double* d_B, double t)
{
	unsigned int line_id_x = threadIdx.x + blockIdx.x * blockDim.x;
	int i = line_id_x / M;
	int j = line_id_x % M;
	if (i < N && j < M)
	{
		if (i > 0 && j > 0 && i < N - 1 && j < M - 1)
		{
			double x = j * hx;
			double y = i * hy;
			d_B(i, j) = d_A(i, j) + tau * (a * a * ((d_A(i, j + 1) - 2 * d_A(i, j) + d_A(i, j - 1)) / (hx * hx) + (d_A(i + 1, j) - 2 * d_A(i, j) + d_A(i - 1, j)) / (hy * hy)) + Q(t, x, y));
		}
	}
}

int main()
{
	double* h_A, * h_B;
	double* d_A, * d_B;
	clock_t t1, t2;
	cudaEvent_t event1, event2;
	float time_device = 0.0f;
	float time_device_copy = 0.0f;
	cudaError_t err;

	// Выделяем память                                    
	h_A = (double*)malloc(N * M * sizeof(double));
	h_B = (double*)malloc(N * M * sizeof(double));

	cudaMalloc((void**)&d_A, N * M * sizeof(double)); CHECK_ERR("malloc_1\n");
	cudaMalloc((void**)&d_B, N * M * sizeof(double)); CHECK_ERR("malloc_1\n");

	cudaEventCreate(&event1);
	cudaEventCreate(&event2);

	U_0(h_A);
	U_0(h_B);
	//print_matrix(h_A, 0);

	t1 = clock();

	double t = 0;

	//print_matrix(h_A, t);

	while (t < T - tau / 2)
	{
		U_n_host(h_A, h_B, t);

		t += tau;

		double* tmp;
		tmp = h_A;
		h_A = h_B;
		h_B = tmp;

		//print_matrix(h_A, t);
	}

	t2 = clock();
	printf("Time work on host %lf msec\n", (double)(t2 - t1) / CLOCKS_PER_SEC * 1000);

	U_0(h_B);

	cudaEventRecord(event1, 0);
	cudaMemcpy(d_A, h_B, N * M * sizeof(double), cudaMemcpyHostToDevice); CHECK_ERR("memcpy_1\n");
	cudaMemcpy(d_B, h_B, N * M * sizeof(double), cudaMemcpyHostToDevice); CHECK_ERR("memcpy_2\n");
	cudaEventRecord(event2, 0);
	cudaEventSynchronize(event2);
	cudaEventElapsedTime(&time_device_copy, event1, event2);
	printf("Time copy to device %lf msec\n", time_device_copy);

	// kernel
	int threads = 32;
	int blocks = (N * M) / threads + 1;

	cudaEventRecord(event1, 0);
	t = 0;

	while (t < T - tau / 2)
	{
		U_n_device << <blocks, threads >> > (d_A, d_B, t); CHECK_ERR("kernel_1\n");

		t += tau;

		double* tmp;
		tmp = d_A;
		d_A = d_B;
		d_B = tmp;
	}
	cudaEventRecord(event2, 0);
	cudaEventSynchronize(event2);
	cudaEventElapsedTime(&time_device, event1, event2);
	printf("Time work on device %lf msec\n", time_device); CHECK_ERR("kernel_15\n");

	cudaEventRecord(event1, 0);
	cudaMemcpy(h_B, d_A, N * M * sizeof(double), cudaMemcpyDeviceToHost); CHECK_ERR("memcpy_3\n");
	cudaEventRecord(event2, 0);
	cudaEventSynchronize(event2);
	cudaEventElapsedTime(&time_device_copy, event1, event2);
	printf("Time copy to host %lf msec\n", time_device_copy);


	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < M; j++)
		{
			if (fabs(h_B(i, j) - h_A(i, j)) > 1e-8)
			{
				printf("ERROR %lf, %lf \n", h_A, h_B);
			}
		}
	}
	print_matrix(h_A, t);
	print_matrix(h_B, t);

	free(h_A); free(h_B);
	cudaFree(d_A); cudaFree(d_B);
}
