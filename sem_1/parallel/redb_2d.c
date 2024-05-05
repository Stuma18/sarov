#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define  Max(a,b) ((a)>(b)?(a):(b))

#define  N   (4096)
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
	time_t t0 = time(0);

	int it;
	init();

	for(it=1; it<=itmax; it++)
	{
		eps = 0.;
		relax();
		printf( "it=%4i   eps=%f\n", it,eps);
		if (eps < maxeps) break;
	}

	verify();

	time_t t1 = time(0);
	double time_in_seconds = difftime(t1, t0);
	printf("%f\n", time_in_seconds);

	return 0;
}


void init()
{ 
	for(j=0; j<=N-1; j++)
	for(i=0; i<=N-1; i++)
	{
		if(i==0 || i==N-1 || j==0 || j==N-1) A[i][j]= 0.;
		else A[i][j]= ( 1. + i + j) ;
	}
} 


void relax()
{

	for(j=1; j<=N-2; j++)
	for(i=1; i<=N-2; i++)
	if ((i + j) % 2 == 1)
	{
		double b;
		b = w*((A[i-1][j]+A[i+1][j]+A[i][j-1]+A[i][j+1])/4. - A[i][j]);
		eps =  Max(fabs(b),eps);
		A[i][j] = A[i][j] + b;
	}
	for(j=1; j<=N-2; j++)
	for(i=1; i<=N-2; i++)
	if ((i + j) % 2 == 0)
	{
		double b;
		b = w*((A[i-1][j]+A[i+1][j]+A[i][j-1]+A[i][j+1])/4. - A[i][j]);
		A[i][j] = A[i][j] + b;
	}

}


void verify()
{ 
	double s;

	s=0.;
	for(j=0; j<=N-1; j++)
	for(i=0; i<=N-1; i++)
	{
		s=s+A[i][j]*(i+1)*(j+1)/(N*N);
	}
	printf("  S = %f\n",s);
}
