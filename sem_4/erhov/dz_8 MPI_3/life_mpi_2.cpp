#include <iostream>
#include <fstream>
#include "mpi.h"
#include <cmath>
#include <time.h>

using namespace std;

int f(int* data, int i, int j, int n)
{
	int state = data[i*(n+2)+j];
	int s = -state;
	for( int ii = i - 1; ii <= i + 1; ii ++ )
		for( int jj = j - 1; jj <= j + 1; jj ++ )
			s += data[ii*(n+2)+jj];
	if( state==0 && s==3 )
		return 1;
	if( state==1 && (s<2 || s>3) ) 
		return 0;
	return state;
}

void update_data(int n, int* data, int* temp)
{
	for( int i=1; i<=n; i++ )
		for( int j=1; j<=n; j++ )
			temp[i*(n+2)+j] = f(data, i, j, n);
}

void init(int n, int* data, int* temp)
{
	for( int i=0; i<(n+2)*(n+2); i++ )
		data[i] = temp[i] = 0;
	int n0 = 1+n/2;
	int m0 = 1+n/2;
	data[(n0-1)*(n+2)+m0] = 1;
	data[n0*(n+2)+m0+1] = 1;
	for( int i=0; i<3; i++ )
		data[(n0+1)*(n+2)+m0+i-1] = 1;
}

void setup_boundaries(int n, int* data)
{
	for( int i=0; i<n+2; i++ )
	{
		data[i*(n+2)+0] = data[i*(n+2)+n];
		data[i*(n+2)+n+1] = data[i*(n+2)+1];
	}
	for( int j=0; j<n+2; j++ )
	{
		data[0*(n+2)+j] = data[n*(n+2)+j];
		data[(n+1)*(n+2)+j] = data[1*(n+2)+j];
	}
}

void setup_boundaries_mpi(int n, int * data, int rank, int p) {
    int i = (rank - 1) / p;
    int j = (rank - 1) % p;
    int left = (j == 0) ? rank + p - 1: rank - 1;
    int right = (j == p - 1) ? rank - p + 1: rank + 1;
    int above = (i == 0) ? p - 1: i - 1;
    
    above = above * p + j + 1;
    int below = (i == p - 1) ? 0: i + 1;
    below = below * p +j + 1;
    MPI_Datatype column;
    MPI_Type_vector(n + 2, 1, n + 2, MPI_INT, &column);
    MPI_Type_commit(&column);
    MPI_Sendrecv(&data[1], 1, column, left, 0, &data[n + 1], 1, column, right, 0, MPI_COMM_WORLD, 0);
    MPI_Sendrecv(&data[n], 1, column, right, 0, &data[0], 1, column, left, 0, MPI_COMM_WORLD, 0);
    MPI_Sendrecv(&data[n + 2], n + 2, MPI_INT, above, 0, &data[(n + 2) * (n + 1)], n + 2, MPI_INT, below, 0, MPI_COMM_WORLD, 0);
    MPI_Sendrecv(&data[(n + 2) * n], n + 2, MPI_INT, below, 0, &data[0], n + 2, MPI_INT, above, 0, MPI_COMM_WORLD, 0);
}

void collectdata(int *data, int n, int p, int rank) {
    if (rank == 0) {
        MPI_Datatype blockrecv;
        int N = n * p;
        MPI_Type_vector(n, n, N + 2, MPI_INT, &blockrecv);
        MPI_Type_commit(&blockrecv);
        for(int i = 0 ; i < p; ++i) {
            for(int j = 0 ; j < p; ++j) {
                MPI_Recv(&(data[(i * p + j) / p * (N + 2) * n + (i * p + j) % p * n  + N + 2 + 1]), 1, blockrecv, i * p + j + 1, 0, MPI_COMM_WORLD, 0);
            }
        }
    } else {
        MPI_Datatype blocksend;
        int N = n * p;
        MPI_Type_vector(n, n, n + 2, MPI_INT, &blocksend);
        MPI_Type_commit(&blocksend);
        MPI_Send(&data[n + 2 + 1], 1, blocksend, 0, 0, MPI_COMM_WORLD);
    }

}

void distribute_data(int *data, int n, int p, int rank) {
    if (rank == 0) {
        MPI_Datatype block;
        int N = n * p;
        MPI_Type_vector(n + 2, n + 2, N + 2, MPI_INT, &block);
        MPI_Type_commit(&block);
        for(int i = 0 ; i < p; ++i) {
            for(int j = 0 ; j < p; ++j) {
                MPI_Send(&(data[(i * p + j) / p * (N + 2) * n + (i * p + j) % p * n  ]), 1, block, i * p + j + 1, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        MPI_Recv(data, (n + 2) * (n + 2), MPI_INT, 0, 0, MPI_COMM_WORLD, 0);
    }
}


void run_life(int n, int T, int rank, int size)
{
	MPI_Comm comm;

	int dims[] = {0, 0};
    MPI_Dims_create(size, 2, dims);

	int periods[2] = {true, true};
	int reorder = true;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &comm);
	

    int left, right, top, bot;
    MPI_Cart_shift(comm, 1, 1, &left, &right);
    MPI_Cart_shift(comm, 0, 1, &top, &bot);

	const int p = dims[0];
	const int N = n*p;

	MPI_Datatype
		global_matrix_not_resized,
		global_matrix,
		matrix_not_resized,
		matrix,
		row,
		column;

	int *data, *temp, *global_data;

	if (rank == 0)
	{
		global_data = new int[N*N];

		init(N-2, global_data);

		int sizes[2]    = {N, N};
		int subsizes[2] = {n, n};
		int starts[2]   = {0, 0};

		MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_INT, &global_matrix_not_resized);  
		MPI_Type_create_resized(global_matrix_not_resized, 0, n*sizeof(int), &global_matrix); // extent = кол-во столбцов матрицы
		MPI_Type_commit(&global_matrix);
	}

	data = new int[(n+2)*(n+2)]; {};
	temp = new int[(n+2)*(n+2)]; {};

	int sizes[2]    = {n+2, n+2};
	int subsizes[2] = {n, n};
	int starts[2]   = {1, 1};

	MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_INT, &matrix);
	MPI_Type_commit(&matrix);

    MPI_Type_contiguous(n+2, MPI_INT, &row);
	MPI_Type_commit(&row);

    MPI_Type_vector(n+2, 1, n+2, MPI_INT, &column);
    MPI_Type_commit(&column);

	int *counts = new int[size];
	int *displs = new int[size];

	int counter = 0;
	for (int i = 0; i < p; i++)
	{
		for (int j = 0; j < p; j++)
		{
			counts[counter] = 1;
			displs[counter] = i*N+j;
			counter++;
		}
	}

	MPI_Scatterv(global_data, counts, displs, global_matrix, data, 1, matrix, 0, MPI_COMM_WORLD);

	double elapsed_time = 0;

	if (!rank)
	{
		elapsed_time = MPI_Wtime();
	}
	
	for( int t = 0; t < T; t++ )
	{
		MPI_Sendrecv(data+n, 1, column, right, 123, data, 1, column, left, 123, comm, MPI_STATUS_IGNORE);
		MPI_Sendrecv(data+1, 1, column, left, 124, data+n+1, 1, column, right, 124, comm, MPI_STATUS_IGNORE);

		MPI_Sendrecv(data+n*(n+2), 1, row, bot, 125, data, 1, row, top, 125, comm, MPI_STATUS_IGNORE);
		MPI_Sendrecv(data+1*(n+2), 1, row, top, 126, data+(n+1)*(n+2), 1, row, bot, 126, comm, MPI_STATUS_IGNORE);

		update_data(n, data, temp);
		swap(data, temp);
	}

	MPI_Gatherv(data, 1, matrix, global_data, counts, displs, global_matrix, 0, MPI_COMM_WORLD);

	if (rank == 0)
	{
		elapsed_time = MPI_Wtime() - elapsed_time;

		ofstream f("output.dat");
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
				f << global_data[i*N+j];
			f << endl;
		}
		f.close();

		ofstream f2("stat.dat");
		f2 << n << " " << T << " " << p << " " << elapsed_time << endl;
		f2.close();

		ofstream f3("data_for_plots.dat", ios::app);
		f3 << N << " " << size << " " << elapsed_time << endl;
		f3.close();
	}

	delete[] counts;
	delete[] displs;

	if (rank == 0)
	{
		delete[] global_data;
		MPI_Type_free(&global_matrix);
	}

	MPI_Type_free(&row);
    MPI_Type_free(&column);
	MPI_Type_free(&matrix);

	delete[] data;
	delete[] temp;
}

int main(int argc, char** argv)
{

    MPI_Init(&argc, &argv);
    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank) ;
    MPI_Comm_size(MPI_COMM_WORLD, &size) ;
    int n = atoi(argv[1]);
    int T = atoi(argv[2]);

    run_life(n, T, rank, size);

    MPI_Finalize();

    return 0;
}


// mpic++ life_mpi.cpp 
// mpiexec -n 1 ./a.out 1000 100
// mpirun -n 1 --oversubscribe ./a.out 150 4800