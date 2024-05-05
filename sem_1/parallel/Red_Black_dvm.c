#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#define  Max(a,b) ((a)>(b)?(a):(b))

#define  N   (4096)
double   maxeps = 0.1e-7;
int itmax = 100;
int i,j,k;
double w = 0.5;
double eps;
double dvtime_();
double time1;

double A [N][N];

void relax();
void init();
void verify();


int main(int an, char **as)
{
	int it;
	init();

    time1 = -dvtime_();

	for (it = 1; it <= itmax; it++)
    {
		eps = 0.;
		#pragma dvm actual (eps)
		
		relax();
		
		#pragma dvm get_actual (eps)
		printf( "it = %4i   eps = %f\n", it, eps);
		
		if (eps < maxeps) break;
	}

	#pragma dvm get_actual(A)
	
	verify();

    time1 += dvtime_();
    printf("%f\n", time1);

	return 0;
}
/*
void init()
{
    #pragma dvm region
    {
        #pragma dvm parallel(2)
		for (int i = 0; i <= N - 1; i++)
		for (int j = 0; j <= N - 1; j++)
    	{
			if (i == 0 || i == N - 1 || j == 0 || j == N - 1) 
				A[i][j] = 0.;
			else 
				A[i][j] = ( 1. + i + j);
		}
	}
}
*/

void init()
{
	for (i = 0; i <= N - 1; i++)
	for (j = 0; j <= N - 1; j++)
	{
		if (i == 0 || i == N - 1 || j == 0 || j == N - 1) 
			A[i][j] = 0.;
		else 
			A[i][j] = ( 1. + i + j);
	}
}


void relax()
{
	#pragma dvm region
	{
		#pragma dvm parallel(2)reduction(max(eps))
		for (int i = 1; i <= N - 2; i++)
		for (int j = 1; j <= N - 2; j++)
		{
			if ((i + j) % 2 == 1)
        	{
				double b;
				b = w * ((A[i - 1][j] + A[i + 1][j] + A[i][j - 1] + A[i][j + 1]) / 4. - A[i][j]);
				eps =  Max(fabs(b),eps);
					A[i][j] = A[i][j] + b;
			}
		}
		
		#pragma dvm parallel(2)
		for (int i = 1; i <= N - 2; i++)
		for (int j = 1; j <= N - 2; j++)
		{
		    if ((i + j) % 2 == 0)
            {
				double b;
				b = w * ((A[i - 1][j] + A[i + 1][j] + A[i][j - 1] + A[i][j + 1]) / 4. - A[i][j]);
					A[i][j] = A[i][j] + b;
			}
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
	for (j = 0; j <= N - 1; j++)
	{
		s = s + A[i][j] * (i + 1) * (j + 1) / (N * N);
	}

	printf("  S = %f\n",s);
}
