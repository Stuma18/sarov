#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define TAG_TOP_TO_BOT 123
#define TAG_BOT_TO_TOP 321

double phi(double x, double y)
{
    return sin(3 * x) * cos(4 * y);
}

double psi_mu_1(double x, double y)
{
    return 0;
}

double f(double a, double t, double x, double y)
{
    return (25 * a * a - 4) * cos(2 * t) * sin(3 * x) * cos(4 * y);
}

double mu_2(double t, double x)
{
    return cos(2 * t) * sin(3 * x);
}

double mu_4(double t, double x, double Ly)
{
    return cos(2 * t) * sin(3 * x) * cos(4 * Ly);
}

int main(int argc, char** argv) 
{
    MPI_Init(&argc, &argv);

	int size, rank;

	MPI_Comm_size(MPI_COMM_WORLD, &size);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    int N, Mx, My;
    N = atoi(argv[1]),
	Mx = atoi(argv[2]),
	My = atoi(argv[3]);
/*
    N = 2000; // размер сетки по t
    Mx = 1000;
    My = 1000; // размер сетки по пространству x и y
*/
    double T, Lx, Ly;
    T = 1; // сетка по времени от [0,T]
    Lx = 1;
    Ly = 1; // сетка по пространству x и y [0,L]

    double tau, hx, hy;
    tau = T / N; // шаг по времени
    hx = Lx / Mx; // шаг по пространству x
    hy = Ly / My; // шаг по пространству y

    //tau = 0.000625;
    //hx = 0.01/2;
    //hy = 0.01/2;

    if(tau < sqrt(2) * hx * hx)
    {
        if(rank == 0)
        {
            printf("ERROR: tau < sqrt(2) * hx\n");
        }
        MPI_Finalize();
        return 1;
    }

    double a = 1;

    double norm_L_2 = 0, norm_c = 0; // нормы
    double norm_L_2_global, norm_c_global;

    const int 
        i_size = (My+1) / size + (rank < (My+1) % size ? 1 : 0),                         // количество строк, которые есть у процесса
        i_size_with_overlap = i_size + (!rank || rank == size-1 ? 1 : 2),                // кол-во строк + теневые строки (по 1 у 0 и последнего и по 2 у всех остальных)
        i_min = rank < (My+1) % size ? i_size * rank : (My+1) - (size-rank)*i_size,      // первая строка, которая есть у процесса
        i_max = i_min + i_size,                                                          // последняя строка, которая есть у процесса
        i_min_with_overlap = i_min - (rank > 0),                                         // теневая строка до
        i_max_with_overlap = i_max + (rank < size-1);                                    // теневая строка после

    double** u_0;
    double** u_1;
    double** u_2;

    // константы для упрощения подсчета формул
    double con_1, con_2, con_3, con_4;
    con_1 = tau * tau / 2 * a * a;
    con_2 = con_1 / (hx * hx);
    con_3 = con_1 * 2;
    con_4 = con_2 * 2;

    u_0 = malloc((i_size_with_overlap + 1) * sizeof(double*));
    u_1 = malloc((i_size_with_overlap + 1) * sizeof(double*));
    u_2 = malloc((i_size_with_overlap + 1) * sizeof(double*));

    for (int i = 0; i <= i_size_with_overlap; ++i)
    {
        u_0[i] = malloc((Mx + 1) * sizeof(double));
        u_1[i] = malloc((Mx + 1) * sizeof(double));
        u_2[i] = malloc((Mx + 1) * sizeof(double));
    }

    double t = 0; // текущее время

    double time1, time2;

    MPI_Barrier(MPI_COMM_WORLD);
    time1 = MPI_Wtime();

    for(int i_2 = i_min_with_overlap; i_2 < i_max_with_overlap; i_2++)
    {
        for(int j = 0; j < Mx + 1; j++)
        {
            const int i = i_2 - i_min_with_overlap;
            u_1[i][j] = phi(j * hx, i_2 * hy);
        }
    }

    for(int i_2 = fmax(1, i_min); i_2 < fmin(My, i_max); i_2++) // все кроме граничных условий
    {
        for(int j = 1; j < Mx; j++)
        {
            const int i = i_2 - i_min_with_overlap;         
//            u_2[i][j] = u_1[i][j] + tau * (psi_mu_1(j * hx, i_2 * hy) + tau / 2 * (a * a * ((u_1[i + 1][j] - 2 * u_1[i][j] + u_1[i - 1][j]) / (hy * hy) + (u_1[i][j + 1] - 2 * u_1[i][j] + u_1[i][j - 1]) / (hx * hx)) + f(a, t, j * hx, i_2 * hy)));
            u_2[i][j] = u_1[i][j] + tau * psi_mu_1(j * hx, i_2 * hy) 
            + con_2 * (u_1[i + 1][j] - 2 * u_1[i][j] + u_1[i - 1][j]
            + u_1[i][j + 1] - 2 * u_1[i][j] + u_1[i][j - 1]) + con_1 * f(a, t, j * hx, i_2 * hy);
        }
    }

    t += tau;

    // граничные условия (для распаралеленого случая)
    // самый левый столбец и самый правый столбец
    if(size != 1)
    {
        for(int i_2 = fmax(1, i_min); i_2 < fmin(My, i_max); i_2++)
        {
            const int i = i_2 - i_min_with_overlap;
            u_2[i][0] = psi_mu_1(t, Lx);
            u_2[i][Mx] = mu_4(t, Lx, i_2 * hy);
        }

        // самая верхняя строка и самая нижняя строка
        if (rank == 0)
        {
            for (int j = 0; j <= Mx; ++j)
            {
                u_2[i_min - i_min_with_overlap][j] = mu_2(t, j * hx);
            }
        }
        else if (rank == size - 1)
        for(int j = 0; j < Mx + 1; j++)
        {
            u_2[i_max - 1 - i_min_with_overlap][j] = mu_4(t, j * hx, Ly);
        }
    }

    // граничные условия (для нераспаралеленого случая)
    if(size == 1)
    {
        // самый левый столбец и самый правый столбец
        for(int i = 0; i < My + 1; i++)
        {
            u_2[i][0] = psi_mu_1(t, Lx);
            u_2[i][Mx] = mu_4(t, Lx, i * hy);
        }
        // самая верхняя строка и самая нижняя строка
        for(int j = 0; j < Mx + 1; j++)
        {
            u_2[0][j] = mu_2(t, j * hx);
            u_2[My][j] = mu_4(t, j * hx, Ly);
        }
    }

    // ищем соседей для процесса
    const int bot = (rank - 1 < 0) ? (MPI_PROC_NULL) : (rank - 1);          // сосед rank - 1
	const int top = (rank + 1 >= size) ? (MPI_PROC_NULL) : (rank + 1);      // сосед rank + 1

    //MPI_Barrier(MPI_COMM_WORLD);
    //printf("rank = %d\n bot = %d\n top = %d\n, 1 = %d\n, 2 = %d\n, 3 = %d\n, 4 = %d\n", rank, bot, top, (i_max - i_min_with_overlap), (i_min_with_overlap - i_min_with_overlap), (i_max-1 - i_min_with_overlap), (i_min - i_min_with_overlap));
    //printf("rank = %d\n, Irecv_1 = %d\n, Irecv_2 = %d\n, Isend_1 = %d\n, Isend_2 = %d\n", rank, u_2[i_max - i_min_with_overlap], u_2[i_min_with_overlap - i_min_with_overlap], u_2[i_max-1 - i_min_with_overlap], u_2[i_min - i_min_with_overlap]);
    //MPI_Barrier(MPI_COMM_WORLD);
    
    while(t < T - tau / 2)
    {
//		MPI_Sendrecv(u_2[i_max-1 - i_min_with_overlap], Mx+1, MPI_DOUBLE, top, TAG_BOT_TO_TOP, u_2[i_min_with_overlap - i_min_with_overlap], Mx+1, MPI_DOUBLE, bot, TAG_BOT_TO_TOP, MPI_COMM_WORLD, &status);
//        MPI_Sendrecv(u_2[i_min - i_min_with_overlap], Mx+1, MPI_DOUBLE, bot, TAG_TOP_TO_BOT, u_2[i_max - i_min_with_overlap], Mx+1, MPI_DOUBLE, top, TAG_TOP_TO_BOT, MPI_COMM_WORLD, &status);

        // асинхронная передача и прием сообщения (пересылки крайних данных, те которые находятся на границах)
        // top и bot - идентификатор получателей и отправителей
        // TAG_TOP_TO_BOT и TAG_BOT_TO_TOP - тег сообщения, чтобы понимать кому сообщение
        MPI_Request request[4];

        MPI_Irecv(u_2[i_max - i_min_with_overlap], Mx+1, MPI_DOUBLE, top, TAG_TOP_TO_BOT, MPI_COMM_WORLD, &request[0]);

        MPI_Irecv(u_2[i_min_with_overlap - i_min_with_overlap], Mx+1, MPI_DOUBLE, bot, TAG_BOT_TO_TOP, MPI_COMM_WORLD, &request[1]);

        MPI_Isend(u_2[i_max-1 - i_min_with_overlap], Mx+1, MPI_DOUBLE, top, TAG_BOT_TO_TOP, MPI_COMM_WORLD, &request[2]);

        MPI_Isend(u_2[i_min - i_min_with_overlap], Mx+1, MPI_DOUBLE, bot, TAG_TOP_TO_BOT, MPI_COMM_WORLD, &request[3]);
        
        MPI_Waitall(4, request, MPI_STATUS_IGNORE);

        double** tmp = u_0;
        u_0 = u_1;
        u_1 = u_2;
        u_2 = tmp;

        for(int i_2 = fmax(1, i_min); i_2 < fmin(My, i_max); i_2++)
        {
            for(int j = 1; j < Mx; j++)
            {
                const int i = i_2 - i_min_with_overlap;
//                u_2[i][j] = 2 * u_1[i][j] - u_0[i][j] + tau * tau * (a * a * ((u_1[i + 1][j] - 2 * u_1[i][j] + u_1[i - 1][j]) / (hy * hy) + (u_1[i][j + 1] - 2 * u_1[i][j] + u_1[i][j - 1]) / (hx * hx)) + f(a, t, j * hx, i_2 * hy));
                u_2[i][j] = 2 * u_1[i][j] - u_0[i][j]
                + con_4 * (u_1[i + 1][j] - 2 * u_1[i][j] + u_1[i - 1][j]
                + u_1[i][j + 1] - 2 * u_1[i][j] + u_1[i][j - 1]) + con_3 * f(a, t, j * hx, i_2 * hy);
            }
        }

        t += tau;

        // граничные условия (для распаралеленого случая)
        // самый левый столбец и самый правый столбец
        if(size != 1)
        {
            for(int i_2 = fmax(1, i_min); i_2 < fmin(My, i_max); i_2++)
            {
                const int i = i_2 - i_min_with_overlap;
                u_2[i][0] = psi_mu_1(t, Lx);
                u_2[i][Mx] = mu_4(t, Lx, i_2 * hy);
            }

            // самая верхняя строка и самая нижняя строка
            if (rank == 0)
            {
                for(int j = 0; j < Mx + 1; j++)
                {
                    u_2[i_min - i_min_with_overlap][j] = mu_2(t, j * hx);
                }
            }
            else if (rank == size - 1)
            {
                for(int j = 0; j < Mx + 1; j++)
                {
                    u_2[i_max-1 - i_min_with_overlap][j] = mu_4(t, j * hx, Ly);
                }
            }
        }

        // граничные условия (для нераспаралеленого случая)
        if(size == 1)
        {
            // самый левый столбец и самый правый столбец
            for(int i = 0; i < My + 1; i++)
            {
                u_2[i][0] = psi_mu_1(t, Lx);
                u_2[i][Mx] = mu_4(t, Lx, i * hy);
            }
            // самая верхняя строка и самая нижняя строка
            for(int j = 0; j < Mx + 1; j++)
            {
                u_2[0][j] = mu_2(t, j * hx);
                u_2[My][j] = mu_4(t, j * hx, Ly);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    time2 = MPI_Wtime();


    // считаем нормы для того, чтобы сравнить с аналитическим решение
    for(int i_2 = i_min; i_2 < i_max; i_2++)
    {
        for(int j = 0; j <= Mx; ++j)
        {
            const int i = i_2 - i_min_with_overlap;
            const double u_t = mu_4(T, j * hx, i_2 * hy) -  u_2[i][j];
            norm_L_2 += u_t * u_t;
            norm_c = fmax(norm_c, fabs(u_t));
        }
    }

    // суммируем все значения norm_L_2 и сохраняем в norm_L_2_global
    MPI_Reduce(&norm_L_2, &norm_L_2_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if (rank == 0)
	{
		norm_L_2_global = sqrt(hx * hy * norm_L_2_global);
	}

    // ищем максимальное значение norm_c и сохраняем в norm_c_global
	MPI_Reduce(&norm_c, &norm_c_global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    for (int i = 0; i < i_size_with_overlap; ++i)
    {
        free(u_0[i]);
        free(u_1[i]);
        free(u_2[i]);
    }

    free(u_0);
    free(u_1);
    free(u_2);

    if (rank == 0)
    {
        printf("%.3le, %.3le, Time = %.5lf\n", norm_L_2_global, norm_c_global, time2 - time1);
    }
    
    MPI_Finalize();
    
    return 0;
}

/*
ssh sarov04@188.44.52.97
Dvg7m

module load openmpi4/openmpi4
module load gcc/gcc-7.4

mpicc str_mpi.c -lm

mpirun -np 8 ./a.out  

mpirun --oversubscribe -np 8 ./a.out
*/

