#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h> 


#define  Max(a,b) ((a)>(b)?(a):(b))

#define  N   (256)
double   maxeps = 0.1e-7;
int itmax = 100;
int i,j,k;
double w = 0.5;
double eps;

double A [N][N];

void relax();
void init();
void verify(); 

int main(int an, char **as)
{

	double time_spent = 0.0;
	clock_t begin = clock();
    
	int it;
	init();

	for(it = 1; it <= itmax; it++)
	{
		eps = 0.;
		relax();
		printf( "it = %4i   eps = %f\n", it, eps);
		if (eps < maxeps) break;
	}

	verify();

	clock_t end = clock();
	time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
	printf("The elapsed time is %f seconds", time_spent);

	return 0;
}


void init()
{ 
	for(i = 0; i <= N - 1; i++)
	for(j = 0; j <= N - 1; j++)
	{
		if(i == 0 || i == N - 1 || j == 0 || j == N - 1) A[i][j] = 0.;
		else A[i][j] = ( 1. + i + j) ;
	}
} 


void relax()
{

	for(i = 1; i <= N - 2; i++)
	for(j = 1; j <= N - 2; j++)
	if ((i + j) % 2 == 1)
	{
		double b;
		b = w * ((A[i - 1][j] + A[i + 1][j] + A[i][j - 1] + A[i][j + 1]) / 4. - A[i][j]);
		eps =  Max(fabs(b),eps);
		A[i][j] = A[i][j] + b;
	}
	for(i = 1; i <= N - 2; i++)
	for(j = 1; j <= N - 2; j++)
	if ((i + j) % 2 == 0)
	{
		double b;
		b = w * ((A[i - 1][j] + A[i + 1][j] + A[i][j - 1] + A[i][j + 1]) / 4. - A[i][j]);
		A[i][j] = A[i][j] + b;
	}

}


void verify()
{ 
	double s;

	s = 0.;
	for(i = 0; i <= N - 1; i++)
	for(j = 0; j <= N - 1; j++)
	{
		s = s + A[i][j] * (i + 1) * (j + 1) / (N * N);
	}
	printf("  S = %f\n",s);
}
