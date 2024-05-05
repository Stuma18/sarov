#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <mpi-ext.h>
#include <setjmp.h>
#include <signal.h>
#include <sys/time.h>
#define Max(a, b) ((a) > (b) ? (a) : (b))

#define N (128)
#define REAL_INDEX(i_aux, startrow) (i_aux + startrow)

#define INTSIZE 4
#define FILESIZE 1000000


#define KILL_PROC 1
int have_been_killed = 0;

#if KILL_PROC != 0
#define KILL_PROC_RANK 1
#endif

MPI_Comm mpi_comm_world_custom;


double maxeps = 0.1e-7;
int itmax = 100;
int i, j;
double w = 0.5;
double eps;

int nprocs, count, nints;

typedef double type_array[N];
type_array * A;

jmp_buf jbuf;

void relax();
void init();
void verify();
void initialize_glob_row_borders(int ranksize, int myrank);
void init_zero();

MPI_Request req[4];
int myrank, ranksize;
int startrow, lastrow, nrows;
MPI_Status status[4];

int ll, shift;
int rc;
void readCP() {
        MPI_Status status;
        MPI_File fh;
        MPI_Offset offset;
        MPI_File_open(mpi_comm_world_custom, "cp_10", MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

        if (myrank!=0) {
                nints = (nrows+2) * N;
                offset = (startrow - 1) * N * sizeof(double);
                MPI_File_read_at(fh, offset, A[0], nints, MPI_DOUBLE, &status);
        } else {
                nints = (nrows+1) * N;
                offset = 0;
                MPI_File_read_at(fh, offset, A[1], nints, MPI_DOUBLE, &status);
        }
        MPI_Get_count(&status, MPI_INT, &count);
        printf("process %d read %d ints\n", myrank, count);
        MPI_File_close(&fh);
}

void writeCP() {
        MPI_Status status;
        MPI_File fh;
        MPI_Offset offset;
        MPI_File_open(mpi_comm_world_custom, "cp_10", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
        nints = nrows * N;
        offset = startrow * N * sizeof(double);
        MPI_File_write_at(fh, offset, A[1], nints, MPI_DOUBLE, &status);
        MPI_Get_count(&status, MPI_INT, &count);
        printf("process %d write %d ints\n", myrank, count);
        MPI_File_close(&fh);
}

static void verbose_errhandler(MPI_Comm *comm, int *err, ...)
{
        have_been_killed = 1;

        int amount_f, len;
        int old_size;
        int old_rank;
        char errstr[MPI_MAX_ERROR_STRING];

        MPI_Group group_f;
        MPI_Group group_norm;
        MPI_Comm_rank(mpi_comm_world_custom, &old_rank);
        MPI_Comm_size(mpi_comm_world_custom, &old_size);
        int *norm_ranks = malloc(sizeof(int) * old_size);
        int *f_ranks = malloc(sizeof(int) * amount_f);
        if (old_rank == 0) 
        {
                MPI_Comm_group(mpi_comm_world_custom, &group_norm);
                MPIX_Comm_failure_ack(mpi_comm_world_custom);
                MPIX_Comm_failure_get_acked(mpi_comm_world_custom, &group_f);
                MPI_Group_size(group_f, &amount_f);
                for (int i = 0; i < amount_f; i++)
                        f_ranks[i] = i;
                MPI_Group_translate_ranks(group_f, amount_f, f_ranks, group_norm, norm_ranks);
        }
        MPI_Error_string(*err, errstr, &len);
        MPIX_Comm_shrink(*comm, &mpi_comm_world_custom);
        MPI_Comm_rank(mpi_comm_world_custom, &myrank);
        MPI_Comm_size(mpi_comm_world_custom, &ranksize);
        MPI_Barrier(mpi_comm_world_custom);
        free(A);
        initialize_glob_row_borders(ranksize, myrank);
        A = malloc((nrows + 2) * sizeof(type_array));
        init_zero();
        readCP();
        MPI_Barrier(mpi_comm_world_custom);
        longjmp(jbuf, 1);
}


void initialize_glob_row_borders(int ranksize, int myrank)
{
        startrow = (myrank * N ) / ranksize;
        lastrow = (((myrank + 1) * N ) / ranksize) - 1;
        if (myrank == ranksize - 1) lastrow = N - 1;
                nrows = lastrow - startrow + 1;
                printf("ranksize: %d, myrank %d: startrow %d, lastrow %d\n",
                        ranksize, myrank, startrow, lastrow);
}

int main(int an, char **as)
{
        if ((rc = MPI_Init(&an, &as))) 
        {
                printf("Ошибка запуска %d, выполнение остановлено\n", rc);
                MPI_Abort(MPI_COMM_WORLD, rc);
                return rc;
        }

        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Comm_size(MPI_COMM_WORLD, &ranksize);

        mpi_comm_world_custom = MPI_COMM_WORLD;

        MPI_Errhandler errh;
        MPI_Comm_create_errhandler(verbose_errhandler, &errh);
        MPI_Comm_set_errhandler(mpi_comm_world_custom, errh);
        MPI_Barrier(mpi_comm_world_custom);

        initialize_glob_row_borders(ranksize, myrank);
        A = malloc((nrows + 2) * sizeof(type_array));

        int it;

        struct timeval start, stop;
        double secs;
        if (!myrank)
        {
                gettimeofday(&start, NULL);
        }

        init();
        //readCP();

        for (it = 1; it <= itmax; it++)
        {
                setjmp(jbuf);
                if (it % 10 == 0)
                {
                        writeCP();
                }
                //writeCP();
                //readCP();
                eps = 0.;
                relax(it);
                if (myrank == 0)
                        printf("it=%4i   eps=%f\n", it, eps);
                if (eps < maxeps)
                        break;
        }
        verify();

        if (!myrank)
        {
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

void init_zero()
{
        for (int i_aux = 0; i_aux < nrows + 2; i_aux++)
        for (j = 0; j <= N - 1; j++)
                A[i_aux][j] = 0.;
}

void relax(int it)
{
        double local_eps = eps;
        for (i = 1; i <= nrows; i++) {
                if (((i == 1) && (myrank == 0)) || ((i == nrows) && (myrank == ranksize - 1))) continue;
                for (int j = 1; j <= N - 2; j++)
                {
                        if ((REAL_INDEX(i, startrow) + j) % 2 == 1) 
                        {
                                float b;
                                b = w * ((A[i - 1][j] + A[i + 1][j] + A[i][j - 1] + A[i][j + 1]) / 4. - A[i][j]);
                                local_eps = Max(fabs(b), local_eps);
                                A[i][j] = A[i][j] + b;
                        }
                }
        }
        if (myrank != 0)
                MPI_Irecv(&A[0], N, MPI_DOUBLE, myrank - 1, 1215, mpi_comm_world_custom, &req[0]);
        if (myrank != ranksize - 1)
                MPI_Isend(&A[nrows], N, MPI_DOUBLE, myrank + 1, 1215, mpi_comm_world_custom, &req[2]);
        if (myrank != ranksize - 1)
                MPI_Irecv(&A[nrows + 1], N, MPI_DOUBLE, myrank + 1, 1216, mpi_comm_world_custom, &req[3]);
        if (myrank != 0)
                MPI_Isend(&A[1], N, MPI_DOUBLE, myrank - 1, 1216, mpi_comm_world_custom, &req[1]);

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
        {
                if (((i == 1) && (myrank == 0)) || ((i == nrows) && (myrank == ranksize - 1))) continue;
                for (int j = 1; j <= N - 2; j++)
                if ((REAL_INDEX(i, startrow) + j) % 2 == 0) 
                {
                        float b;
                        b = w * ((A[i - 1][j] + A[i + 1][j] + A[i][j - 1] + A[i][j + 1]) / 4. - A[i][j]);
                        A[i][j] = A[i][j] + b;
                }
        }
        if (myrank != 0)
                MPI_Irecv(&A[0], N , MPI_DOUBLE, myrank - 1, 1215, mpi_comm_world_custom, &req[0]);
        if (myrank != ranksize - 1)
                MPI_Isend(&A[nrows], N , MPI_DOUBLE, myrank + 1, 1215, mpi_comm_world_custom, &req[2]);
        if (myrank != ranksize - 1)
                MPI_Irecv(&A[nrows + 1], N, MPI_DOUBLE, myrank + 1, 1216, mpi_comm_world_custom, &req[3]);
        if (myrank != 0)
                MPI_Isend(&A[1], N, MPI_DOUBLE, myrank - 1, 1216, mpi_comm_world_custom, &req[1]);

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

        #if KILL_PROC != 0
                if ((have_been_killed == 0) && (myrank == KILL_PROC_RANK) && (it == 11)) 
                {
                        printf("KILL \n");
                        raise(SIGKILL);
                }
        #endif

        MPI_Allreduce(&local_eps, &eps, 1, MPI_DOUBLE, MPI_MAX, mpi_comm_world_custom);

}

void verify()
{
        double s = 0, local_s = 0;

        for (i = 1; i <= nrows; i++)
        for (j = 0; j <= N - 1; j++)
        {
                local_s += A[i][j] * (REAL_INDEX(i, startrow) + 1) * (j + 1) / (N * N);
        }


        MPI_Allreduce(&local_s, &s, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_world_custom);
        if (myrank == 0)
                printf("  S = %f\n", s);

}

// mpicc rb_mpi_kill.c
// mpirun -np 4 --with-ft ulfm --mca mpi_ft_detector true --oversubscribe ./a.out