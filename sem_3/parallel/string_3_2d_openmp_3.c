#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <omp.h>

double phi(double x, double y)
{
    return sin(3 * x) * cos(4 * y);
}

double psi(double x, double y)
{
    return 0;
}

double f(double a, double t, double x, double y)
{
    return (25 * a * a - 4) * cos(2 * t) * sin(3 * x) * cos(4 * y);
}

double mu_1(double t)
{
    return 0;
}

double mu_2(double t, double x)
{
    return cos(2 * t) * sin(3 * x);
}

double mu_3(double t, double Lx, double y)
{
    return cos(2 * t) * sin(3 * Lx) * cos(4 * y);
}

double mu_4(double t, double x, double Ly)
{
    return cos(2 * t) * sin(3 * x) * cos(4 * Ly);
}

double f_u_t(double t, double x, double y)
{
    return cos(2 * t) * sin(3 * x) * cos(4 * y);
}

int main(int argc, char** argv) {
    double time1 = omp_get_wtime();

    int N, Mx, My;
    N = atoi(argv[1]),
	Mx = atoi(argv[2]),
	My = atoi(argv[3]);
    //int N, Mx, My;
    //N = 1600; // размер сетки по t
    //Mx = My = 800; // размер сетки по пространству x и y

    double T, Lx, Ly;
    T = 1; // сетка по времени от [0,T]
    Lx = Ly = 1; // сетка по пространству x и y [0,L]

    double tau, hx, hy;
    tau = T / N; // шаг по времени
    hx = Lx / Mx; // шаг по пространству x
    hy = Ly / My; // шаг по пространству y
    //tau = 0.000078125;
    //hx = 0.00125;
    //hy = 0.00125;

    double a = 1;

    double norm_L_2 = 0, norm_c = 0; // нормы

    double** u_0;
    double** u_1;
    double** u_2;
    double** u_t;

    #pragma omp parallel
    {

        #pragma omp single
        u_0 = malloc((My + 1) * sizeof(double*)); // массив значений u в момент времени 0
        u_1 = malloc((My + 1) * sizeof(double*)); // массив значений u в момент времени t_1
        u_2 = malloc((My + 1) * sizeof(double*)); // массив значений u в момент времени t_2
        u_t = malloc((My + 1) * sizeof(double*)); // массив значений u в момент времени T по точному решению
        
        #pragma omp barrier

        #pragma omp for
        for(int i = 0; i <= My; i++)
        {
            u_0[i] = malloc((Mx + 1) * sizeof(double));
            u_1[i] = malloc((Mx + 1) * sizeof(double));
            u_2[i] = malloc((Mx + 1) * sizeof(double));
            u_t[i] = malloc((Mx + 1) * sizeof(double));
        }

        double t = 0; // текущее время

        #pragma omp for
        for(int i = 0; i < My + 1; i++)
        {
            for(int j = 0; j < Mx + 1; j++)
            {
                u_0[i][j] = phi(j * hx, i * hy);
            }
        }

        #pragma omp for
        for(int i = 1; i < My; i++) // все кроме граничных условий
        {
            for(int j = 1; j < Mx; j++)
            {
                u_1[i][j] = u_0[i][j] + tau * (psi(j * hx, i * hy) + tau / 2 * (a * a * 
                ((u_0[i + 1][j] - 2 * u_0[i][j] + u_0[i - 1][j]) / (hy * hy) + (u_0[i][j + 1] - 2 * u_0[i][j] 
                + u_0[i][j - 1]) / (hx * hx)) + f(a, t, j * hx, i * hy)));
            }
        }

        t += tau;

        // граничные условия
        // самый левый столбец и самый правый столбец
        #pragma omp for
        for(int i = 0; i < My + 1; i++)
        {
            u_1[i][0] = mu_1(t);
            u_1[i][Mx] = mu_3(t, Lx, i * hy);
        }

        // самая верхняя строка и самая нижняя строка
        #pragma omp for
        for(int j = 0; j < Mx + 1; j++)
        {
            u_1[0][j] = mu_2(t, j * hx);
            u_1[My][j] = mu_4(t, j * hx, Ly);
        }


        while(t < T - tau / 2)
        {
            #pragma omp for
            for(int i = 1; i < My; i++)
            {
                for(int j = 1; j < Mx; j++)
                {
                u_2[i][j] = 2 * u_1[i][j] - u_0[i][j]
                    + tau * tau * (a * a * ((u_1[i + 1][j] - 2 * u_1[i][j] + u_1[i - 1][j]) / (hy * hy)
                    + (u_1[i][j + 1] - 2 * u_1[i][j] + u_1[i][j - 1]) / (hx * hx)) + f(a, t, j * hx, i * hy));
                }
            }

            t += tau;

            // граничные условия
            // самый левый столбец и самый правый столбец
            #pragma omp for
            for(int i = 0; i < My + 1; i++)
            {
                u_2[i][0] = mu_1(t);
                u_2[i][Mx] = mu_3(t, Lx, i * hy);
            }

            // самая верхняя строка и самая нижняя строка
            #pragma omp for
            for(int j = 0; j < Mx + 1; j++)
            {
                u_2[0][j] = mu_2(t, j * hx);
                u_2[My][j] = mu_4(t, j * hx, Ly);
            }

            #pragma omp single
            {
                double** tmp = u_0;
                u_0 = u_1;
                u_1 = u_2;
                u_2 = tmp;
            }
        }


        // считаем норму
        #pragma omp for reduction(+ : norm_L_2) reduction(max : norm_c)
        /*
        - `reduction(+ : norm_L_2)` указывает, что переменная `norm_L_2` должна быть суммирована 
        в конце каждой итерации цикла. Каждый поток сохраняет свою собственную приватную копию `norm_L_2`, 
        а затем все копии суммируются вместе в конце цикла.

        - `reduction(max : norm_c)` указывает, что переменная `norm_c` должна быть сокращена до максимального значения 
        в конце каждой итерации цикла. Каждый поток хранит свое собственное приватное копия `norm_c`, 
        а затем все копии сравниваются между собой, и максимальное значение сохраняется в `norm_c` в конце цикла.
        */
        for(int i = 0; i <= My; i++)
        {
            for(int j = 0; j <= Mx; j++)
            {
                u_t[i][j] = f_u_t(T, j * hx, i * hy); // считаем нормы для того, чтобы сравнить с аналитическим решение

                norm_L_2 += (u_1[i][j] - u_t[i][j]) * (u_1[i][j] - u_t[i][j]);
                norm_c = fmax(norm_c, fabs(u_1[i][j] - u_t[i][j]));
            }
        }

        #pragma omp single
        norm_L_2 = sqrt(hx * hy * norm_L_2);

    }

    printf("%.3le, %.3le, Time = %.5lf\n", norm_L_2, norm_c, omp_get_wtime() - time1);
}

