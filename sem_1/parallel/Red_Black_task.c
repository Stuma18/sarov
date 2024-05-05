#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h> 
#include <omp.h>


#define  Max(a,b) ((a)>(b)?(a):(b))

#define  N   (128)
double   maxeps = 0.1e-7;
int itmax = 100;
double w = 0.5;
double eps;
double S = 0;

double A [N][N];

void relax1(int start, int count);
void relax2(int start, int count);
void init(int start, int count);
void verify(int start, int count);

int main()
{
    double time1 = omp_get_wtime();
    #pragma omp parallel
    {
		#pragma omp single
		{
			int num_thr = omp_get_num_threads();
			int block_size = N / num_thr;
			int block_last = block_size + N % num_thr;
            int bias = 0;
            int k;
			for (int i = 0; i < num_thr; ++i)
			{
                k = (block_size + bias < block_last? 1 : 0);
            	#pragma omp task
            	init(block_size * i + bias, block_size + k);
                bias += k;
			}
        }

        for(int it = 1; it <= itmax; it++)
        {
            #pragma omp single
            {
		        eps = 0.;
		        int num_thr = omp_get_num_threads();
                int block_size = (N - 2) / num_thr;
                int block_last = block_size + (N - 2) % num_thr;
                int bias = 0;
                int k;
                for (int i = 0; i < num_thr; ++i)
                {
                    k = (block_size + bias < block_last? 1 : 0);
                    #pragma omp task
                	relax1(1 + block_size * i + bias, block_size + k);
                    bias += k;
                }
            }

            #pragma omp single
            {
		        int num_thr = omp_get_num_threads();
                int block_size = (N - 2) / num_thr;
                int block_last = block_size + (N - 2) % num_thr;
                int bias = 0;
                int k;
                for (int i = 0; i < num_thr; ++i)
                {
                    k = (block_size + bias < block_last? 1 : 0);
                    #pragma omp task
                	relax2(1 + block_size * i + bias, block_size + k);
                    bias += k;
                }
            }

			#pragma omp single
        	printf( "it = %4i   eps = %f\n", it, eps);
            if (eps < maxeps) break;
			#pragma omp barrier
        }

		#pragma omp single
        {
            int num_thr = omp_get_num_threads();
            int block_size = N / num_thr;
            int block_last = block_size + N % num_thr;
            int bias = 0;
            int k;
            for (int i = 0; i < num_thr; ++i)
            {
                k = (block_size + bias < block_last? 1 : 0);
                #pragma omp task
                verify(block_size * i + bias, block_size + k);
                bias += k;
            }
        }
        #pragma omp single
        printf("  S = %lf\n", S);	
	}
    double time2 = omp_get_wtime();
    printf("%f\n", time2 - time1);
    return 0;
}

void init(int start, int count)
{ 
	for(int i = start; i < start + count; i++)
	for(int j = 0; j <= N - 1; j++)
	{
		if(i == 0 || i == N - 1 || j == 0 || j == N - 1) A[i][j] = 0.;
		else A[i][j] = ( 1. + i + j);
	}
} 

void relax1(int start, int count)
{
    double local_eps = 0.0;
	for(int i = start; i < start + count; i++)
	for(int j = 1; j <= N - 2; j++)
	if ((i + j) % 2 == 1)
	{
		double b;
		b = w * ((A[i - 1][j] + A[i + 1][j] + A[i][j - 1] + A[i][j + 1]) / 4. - A[i][j]);
		local_eps = Max(fabs(b),local_eps);
		A[i][j] = A[i][j] + b;
	}
	#pragma omp critical
	eps = Max(eps, local_eps);
}

void relax2(int start, int count)
{
	for(int i = start; i < start + count; i++)
	for(int j = 1; j <= N - 2; j++)
	if ((i + j) % 2 == 0)
	{
		double b;
		b = w * ((A[i - 1][j] + A[i + 1][j] + A[i][j - 1] + A[i][j + 1]) / 4. - A[i][j]);
		A[i][j] = A[i][j] + b;
	}
}

void verify(int start, int count)
{ 
	double s = 0.;
    for(int i = start; i < start + count; ++i)
    for(int j = 0; j <= N - 1; j++)
	{
		s = s + A[i][j] * (i + 1) * (j + 1) / (N * N);
	}
    #pragma omp critical
	S += s;
}
