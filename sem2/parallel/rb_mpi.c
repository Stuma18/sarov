#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <sys/time.h>
#define Max(a, b) ((a) > (b) ? (a) : (b))

#define N (128)
#define REAL_INDEX(i_aux, startrow) (i_aux + startrow)

double maxeps = 0.1e-7;
int itmax = 100;
int i, j;
double w = 0.5;
double eps;

typedef double type_array[N];
type_array * A;

void relax();
void init();
void verify();

MPI_Request req[4];
int myrank, ranksize;
int startrow, lastrow, nrows;
MPI_Status status[4];

int ll, shift;

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &ranksize);
	MPI_Barrier(MPI_COMM_WORLD);

	startrow = (myrank * (N - 2)) / ranksize;
	lastrow = (((myrank + 1) * (N - 2)) / ranksize) - 1;
	nrows = lastrow - startrow + 1;

	A = malloc((nrows + 2) * sizeof(type_array));

	int it;

    struct timeval start, stop;
    double secs;
    if (!myrank){
        gettimeofday(&start, NULL);
    }

	init();

	for (it = 1; it <= itmax; it++)
	{
		eps = 0.;
		relax();
		if (myrank == 0)
			printf("it=%4i   eps=%f\n", it, eps);
		if (eps < maxeps)
			break;
	}

	verify();

        if (!myrank){
        gettimeofday(&stop, NULL);
        secs = (double)(stop.tv_usec - start.tv_usec) / 1000000 + (double)(stop.tv_sec - start.tv_sec);
        printf("time taken for thread=%d, N=%d: %f seconds\n", ranksize, N, secs);
    }

	MPI_Finalize();
	return 0;
}

void init()
{
	for (int i_aux = 0; i_aux < nrows + 2; i_aux++)
		for (j = 0; j <= N - 1; j++)
			{
				i = REAL_INDEX(i_aux, startrow);
				if (i == 0 || i == N - 1 || j == 0 || j == N - 1)
					A[i_aux][j] = 0.;
				else
					A[i_aux][j] = (1. + i + j);
			}
}

void relax()
{
	double local_eps = eps;
	for (i = 1; i <= nrows; i++)
		for (int j = 1; j <= N - 2; j++){
            if ((REAL_INDEX(i, startrow) + j) % 2 == 1) {
                float b;
                b = w * ((A[i - 1][j] + A[i + 1][j] + A[i][j - 1] + A[i][j + 1]) / 4. - A[i][j]);
                local_eps = Max(fabs(b), local_eps);
                A[i][j] = A[i][j] + b;
                }
            }

	if (myrank != 0)
		MPI_Irecv(&A[0], N, MPI_DOUBLE, myrank - 1, 1215, MPI_COMM_WORLD, &req[0]);
	if (myrank != ranksize - 1)
		MPI_Isend(&A[nrows], N, MPI_DOUBLE, myrank + 1, 1215, MPI_COMM_WORLD, &req[2]);
	if (myrank != ranksize - 1)
		MPI_Irecv(&A[nrows + 1], N, MPI_DOUBLE, myrank + 1, 1216, MPI_COMM_WORLD, &req[3]);
	if (myrank != 0)
		MPI_Isend(&A[1], N, MPI_DOUBLE, myrank - 1, 1216, MPI_COMM_WORLD, &req[1]);

	ll = 4, shift = 0;
	if (myrank == 0)
	{
		ll -= 2;
		shift = 2;
	}
	if (myrank == ranksize - 1)
	{
		ll -= 2;
	}

	MPI_Waitall(ll, &req[shift], &status[0]);

	for (int i = 1; i <= nrows; i++)
		for (int j = 1; j <= N - 2; j++)
            if ((REAL_INDEX(i, startrow) + j) % 2 == 0) {
                float b;
                b = w * ((A[i - 1][j] + A[i + 1][j] + A[i][j - 1] + A[i][j + 1]) / 4. - A[i][j]);
                A[i][j] = A[i][j] + b;
            }

	if (myrank != 0)
		MPI_Irecv(&A[0], N , MPI_DOUBLE, myrank - 1, 1215, MPI_COMM_WORLD, &req[0]);
	if (myrank != ranksize - 1)
		MPI_Isend(&A[nrows], N , MPI_DOUBLE, myrank + 1, 1215, MPI_COMM_WORLD, &req[2]);
	if (myrank != ranksize - 1)
		MPI_Irecv(&A[nrows + 1], N, MPI_DOUBLE, myrank + 1, 1216, MPI_COMM_WORLD, &req[3]);
	if (myrank != 0)
		MPI_Isend(&A[1], N, MPI_DOUBLE, myrank - 1, 1216, MPI_COMM_WORLD, &req[1]);

	ll = 4, shift = 0;
	if (myrank == 0)
	{
		ll -= 2;
		shift = 2;
	}
	if (myrank == ranksize - 1)
	{
		ll -= 2;
	}

	MPI_Waitall(ll, &req[shift], &status[0]);

	MPI_Allreduce(&local_eps, &eps, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
}

void verify()
{
	double s = 0, local_s = 0;

    for (i = 1; i <= nrows; i++)
        for (j = 0; j <= N - 1; j++)
        {
            local_s += A[i][j] * (REAL_INDEX(i, startrow) + 1) * (j + 1) / (N * N);
        }


	MPI_Allreduce(&local_s, &s, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	if (myrank == 0)
		printf("  S = %f\n", s);
}
