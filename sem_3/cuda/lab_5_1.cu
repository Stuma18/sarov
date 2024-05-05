#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 11							// размер сетки
#define M 11
#define hx 1.0 / (N - 1)				// шаг по сетке
#define hy 1.0 / (M - 1)
#define T 0.5							// время до которого считаем
#define a 1.0							// константа
#define tau	0.25 * h * h / a			// шаг по времени
#define Q(t, x, y) 0.0					// константа
#define eps 0.000001						// точность
#define h_A(i, j) h_A[(i) * M + (j)]
#define h_B(i, j) h_B[(i) * M + (j)]
#define d_A(i, j) d_A[(i) * M + (j)]
#define d_B(i, j) d_B[(i) * M + (j)]
#define d_C(i, j) d_C[(i) * M + (j)]

#ifdef _DEBUG

#define CHECK_ERR(a) { err = cudaGetLastError(); \
				 if(err != cudaSuccess) { printf(a); printf("%s(%d): %s \n", __FILE__, __LINE__, cudaGetErrorString(err)); } }
#else
#define CHECK_ERR(a)
#endif

void U_0(double* h_A)
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < M; j++)
		{
			double x = j * hy;
			double y = i * hy;
			if (j == M - 1 || i == N - 1)
				h_A(i, j) = 1;
			else if (j == 0)
				h_A(i, j) = exp(double(1 - y));
			else if (i == 0)
				h_A(i, j) = exp(double(1 - x));
			else
				h_A(i, j) = 0;
		}
	}
}

/*
void U_0(double* h_A)
{
	int i = 0;
	while (i <= 1)
	{
		int j = 0;
		while (j <= 1)
		{
			if (i == 1 || j == 1)
				h_A(i, j) = 1;
			else if (j == 0)
			{
				h_A(i, j) = exp(double(1 - i));
			}
			else if (i == 0)
			{
				h_A(i, j) = exp(double(1 - j));
			}
			else
				h_A(i, j) = 0;
			j = j + 1 / N;
		}
		i = i + 1 / N;
	}
}*/

void U_n_host(double* h_A, double* h_B, double t)
{
	for (int i = 1; i < N - 1; i++)
	{
		for (int j = 1; j < M - 1; j++)
		{
			double x = j * hy;
			double y = i * hx;
			//h_B(i, j) = h_A(i, j) + tau * (a * a * ((h_A(i, j + 1) - 2 * h_A(i, j) + h_A(i, j - 1)) / (h * h) + (h_A(i + 1, j) - 2 * h_A(i, j) + h_A(i - 1, j)) / (h * h)) + Q(t, x, y));
			h_B(i, j) = ((h_A(i - 1, j) + h_A(i + 1, j)) / (hx * hx) + (h_A(i, j - 1) + h_A(i, j + 1)) / (hy * hy)) / (2 * (1 / (hx * hx) + 1 / (hy * hy)));
		}
	}
}

void print_matrix(double* h_A, double t)
{
	//printf("t = %lf \n", t);
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


__global__ void U_n_device(double* d_A, double* d_B, double* d_C, double t)
{
	unsigned int line_id = threadIdx.x + blockIdx.x * blockDim.x;
	int i = line_id / N;
	int j = line_id % N;
	if (i < N && j < M)
	{
		if (i > 0 && j > 0 && i < N - 1 && j < M - 1)
		{
			double x = j * hy;
			double y = i * hx;
			//d_B(i, j) = d_A(i, j) + tau * (a * a * ((d_A(i, j + 1) - 2 * d_A(i, j) + d_A(i, j - 1)) / (h * h) + (d_A(i + 1, j) - 2 * d_A(i, j) + d_A(i - 1, j)) / (h * h)) + Q(t, x, y));
			d_B(i, j) = ((d_A(i - 1, j) + d_A(i + 1, j)) / (hx * hx) + (d_A(i, j - 1) + d_A(i, j + 1)) / (hy * hy)) / (2 * (1 / (hx * hx) + 1 / (hy * hy)));
			d_C(i, j) = d_B(i, j) - d_A(i, j);
		}
	}
}

int main()
{
	double* h_A, * h_B;
	double* d_A, * d_B, *d_C;
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
	cudaMalloc((void**)&d_C, N * M * sizeof(double)); CHECK_ERR("malloc_1\n");

	cudaEventCreate(&event1);
	cudaEventCreate(&event2);

	U_0(h_A);
	U_0(h_B);
	printf("U_0(h_A)\n");
	print_matrix(h_A, 0); 
	printf("U_0(h_B)\n");
	print_matrix(h_B, 0);

	t1 = clock();

	double t = 0;

	//print_matrix(h_A, t);
	double max = 10.0;
	int k = 0;
	while (max >= eps)
	{
		U_n_host(h_A, h_B, t);

		double* tmp;
		tmp = h_A;
		h_A = h_B;
		h_B = tmp;

		max = 0.0;
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < M; j++)
			{
				max = fmax(max, fabs(h_B(i, j) - h_A(i, j)));
			}
		}
		k++;
		//print_matrix(h_A, t);
	}
	//printf("h_B\n");
	//printf("%d\n", k);
	//print_matrix(h_A, t);

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

/*
	int k_2 = 0;
	//while (max >= eps)
	while (k_2 < k)
	{
		U_n_device << <blocks, threads >> > (d_A, d_B, t); CHECK_ERR("kernel_1\n");

		double* tmp;
		tmp = d_A;
		d_A = d_B;
		d_B = tmp;

		k_2++;
	}
	printf("%d\n", k_2);*/

	int k_2 = 0;
	max = 10.0;
	while (max >= eps)
	{
		U_n_device << <blocks, threads >> > (d_A, d_B, d_C, t); CHECK_ERR("kernel_1\n");

		double* tmp;
		tmp = d_A;
		d_A = d_B;
		d_B = tmp;

		max = 0.0;

		thrust::device_ptr<double> d_ptr = thrust::device_pointer_cast(d_C);
		max = *(thrust::max_element(d_ptr, d_ptr + N * M));

		k_2++;
	}
	//printf("%d\n", k_2);

	cudaEventRecord(event2, 0);
	cudaEventSynchronize(event2);
	cudaEventElapsedTime(&time_device, event1, event2);
	printf("Time work on device %lf msec\n", time_device); CHECK_ERR("kernel_15\n");

	cudaEventRecord(event1, 0);
	cudaMemcpy(h_B, d_A, N * N * sizeof(double), cudaMemcpyDeviceToHost); CHECK_ERR("memcpy_3\n");
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
	printf("%d\n", k);
	printf("h_B\n");
	print_matrix(h_A, t);
	printf("%d\n", k_2);
	printf("d_B\n");
	print_matrix(h_B, t);

	free(h_A); free(h_B);
	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}
