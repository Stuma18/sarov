#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 11							// ������ �����
#define h 1.0 / (N - 1)					// ��� �� �����
#define T 0.5							// ����� �� �������� �������
#define a 1.0							// ���������
#define tau	0.25 * h * h / a			// ��� �� �������
#define Q(t, x, y) 0.0					// ���������
//#define T tau	
#define h_A(i, j) h_A[(i) * N + (j)]
#define h_B(i, j) h_B[(i) * N + (j)]
#define d_A(i, j) d_A[(i) * N + (j)]
#define d_B(i, j) d_B[(i) * N + (j)]

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
		for (int j = 0; j < N; j++)
		{
			double x = j * h;
			double y = i * h;
			if (fabs(x - 0.5) < h / 2 && fabs(y - 0.5) < h / 2)
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
		for (int j = 1; j < N - 1; j++)
		{
			double x = j * h;
			double y = i * h;
			h_B(i, j) = h_A(i, j) + tau * (a * a * ((h_A(i, j + 1) - 2 * h_A(i, j) + h_A(i, j - 1)) / (h * h) + (h_A(i + 1, j) - 2 * h_A(i, j) + h_A(i - 1, j)) / (h * h)) + Q(t, x, y));
		}
	}
}

void print_matrix(double* h_A, double t)
{
	printf("t = %lf \n", t);
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			printf("%.3lf ", h_A(i, j));
		}
		printf("\n");
	}
	printf("\n");
}


__global__ void U_n_device(double* d_A, double* d_B, double t)
{
	unsigned int line_id = threadIdx.x + blockIdx.x * blockDim.x;
	int i = line_id / N;
	int j = line_id % N;
	if (i < N && j < N)
	{
		if (i > 0 && j > 0 && i < N - 1 && j < N - 1)
		{
			double x = j * h;
			double y = i * h;
			d_B(i, j) = d_A(i, j) + tau * (a * a * ((d_A(i, j + 1) - 2 * d_A(i, j) + d_A(i, j - 1)) / (h * h) + (d_A(i + 1, j) - 2 * d_A(i, j) + d_A(i - 1, j)) / (h * h)) + Q(t, x, y));
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

	// �������� ������                                    
	h_A = (double*)malloc(N * N * sizeof(double));
	h_B = (double*)malloc(N * N * sizeof(double));

	cudaMalloc((void**)&d_A, N * N * sizeof(double)); CHECK_ERR("malloc_1\n");
	cudaMalloc((void**)&d_B, N * N * sizeof(double)); CHECK_ERR("malloc_1\n");

	cudaEventCreate(&event1);
	cudaEventCreate(&event2);

	U_0(h_A);
	U_0(h_B);
	//print_matrix(h_A, 0); 
	//print_matrix(h_B, 0);

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
	cudaMemcpy(d_A, h_B, N * N * sizeof(double), cudaMemcpyHostToDevice); CHECK_ERR("memcpy_1\n");
	cudaMemcpy(d_B, h_B, N * N * sizeof(double), cudaMemcpyHostToDevice); CHECK_ERR("memcpy_2\n");
	cudaEventRecord(event2, 0);
	cudaEventSynchronize(event2);
	cudaEventElapsedTime(&time_device_copy, event1, event2);
	printf("Time copy to device %lf msec\n", time_device_copy);

	// kernel
	int threads = 32;
	int blocks = (N * N) / threads + 1;

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
	cudaMemcpy(h_B, d_A, N * N * sizeof(double), cudaMemcpyDeviceToHost); CHECK_ERR("memcpy_3\n");
	cudaEventRecord(event2, 0);
	cudaEventSynchronize(event2);
	cudaEventElapsedTime(&time_device_copy, event1, event2);
	printf("Time copy to host %lf msec\n", time_device_copy);


	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			if (fabs(h_B(i, j) - h_A(i, j)) > 1e-8)
			{
				printf("ERROR %lf, %lf \n", h_A, h_B);
			}
		}
	}
	//print_matrix(h_A, t);
	//print_matrix(h_B, t);

	free(h_A); free(h_B);
	cudaFree(d_A); cudaFree(d_B);
}
