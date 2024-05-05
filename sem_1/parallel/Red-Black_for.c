#include <math.h>
#include <stdlib.h>
#include <stdio.h>
//#include <string.h>
#include <sys/time.h>
#include <omp.h>
#define  Max(a,b) ((a)>(b)?(a):(b))

#define  N   (1024)
double   maxeps = 0.1e-7;
int itmax = 100;
//int i,j,k;
double w = 0.5;
double eps;

double A [N][N];

void relax();
void init();
void verify();


int main(int argc, char **argv)
{
    	//if (argc < 2) {
	//	printf("Bad arguments!\n");
	//	return -1;
	//}

	//int num_thr = atoi(argv[1]);
	int it;
	//omp_set_num_threads(num_thr);
	
	init();

	double time1 = omp_get_wtime();

	for (it = 1; it <= itmax; it++)
    {
		eps = 0.;
		relax();
		printf( "it = %4i   eps = %f\n", it, eps);
		if (eps < maxeps) break;
	}

	double time2 = omp_get_wtime();

	verify();

	printf("%f\n", time2 - time1);

	return 0;
}

void init()
{
    int i;
    int j;
    #pragma omp parallel shared(A)
    {
		#pragma omp for private(i,j)
		for (i = 0; i <= N - 1; i++)
		{
			for (j = 0; j <= N - 1; j++)
        	{
				if (i == 0 || i == N - 1 || j == 0 || j == N - 1) 
					A[i][j] = 0.;
				else 
					A[i][j] = ( 1. + i + j);
			}
		}
	}
}


void relax()
{
	#pragma omp parallel shared(A)
	{
        int i;
        int j;
		double local_eps = eps;
		#pragma omp for private(i, j)
		for (i = 1; i <= N - 2; i++)
		{	
			for (j = 1 + i % 2; j <= N - 2; j += 2)
			{
				if ((i + j) % 2 == 1)
            	{
					double b;
					b = w * ((A[i - 1][j] + A[i + 1][j] + A[i][j - 1] + A[i][j + 1]) / 4. - A[i][j]);
					local_eps = Max(fabs(b),local_eps);
						A[i][j] = A[i][j] + b;
				}
			}
		}
		
		#pragma omp for private(i, j)
		for (i = 1; i <= N - 2; i++)
		{
			for (j = 1 + (i + 1) % 2; j <= N - 2; j += 2)
			{
				if ((i + j) % 2 == 0)
            	{
					double b;
					b = w * ((A[i - 1][j] + A[i + 1][j] + A[i][j - 1] + A[i][j + 1]) / 4. - A[i][j]);
						A[i][j] = A[i][j] + b;
				}
			}
		}
		
		#pragma omp critical
		{
			eps = Max(eps, local_eps);
		}
    }
}


void verify()
{ 
	double s;
	s = 0.;
    int i;
    int j;
	for (i = 0; i <= N - 1; i++)
	{
		for (j = 0; j <= N - 1; j++)
		{
			s = s + A[i][j] * (i + 1) * (j + 1) / (N * N);
		}
	}
	printf("  S = %f\n",s);
}
