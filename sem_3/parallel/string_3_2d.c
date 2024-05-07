#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>

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

int main() {
    //double time1 = omp_get_wtime();

    clock_t start_time = clock();

    int N, Mx, My;
    N = 400; // размер сетки по t
    Mx = My = 200; // размер сетки по пространству x и y

    double T, Lx, Ly;
    T = 1; // сетка по времени от [0,T]
    Lx = Ly = 1; // сетка по пространству x и y [0,L]

    double tau, hx, hy;
    tau = T / N; // шаг по времени
    hx = Lx / Mx; // шаг по пространству x
    hy = Ly / My; // шаг по пространству y

    double a = 1;

    double** u_0 = malloc((My + 1) * sizeof(double*)); // массив значений u в момент времени 0
    for(int i = 0; i <= My; i++)
    {
        u_0[i] = malloc((Mx + 1) * sizeof(double));
    }

    for(int i = 0; i < My + 1; i++)
    {
        for(int j = 0; j < Mx + 1; j++)
        {
            u_0[i][j] = phi(j * hx, i * hy);
        }
    }

    double** u_1 = malloc((My + 1) * sizeof(double*)); // массив значений u в момент времени t_1
    for(int i = 0; i <= My; i++)
    {
        u_1[i] = malloc((Mx + 1) * sizeof(double));
    }

    double t = 0; // текущее время

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
    for(int i = 0; i < My + 1; i++)
    {
        u_1[i][0] = mu_1(t);
        u_1[i][Mx] = mu_3(t, Lx, i * hy);
    }

    // самая верхняя строка и самая нижняя строка
    for(int j = 0; j < Mx + 1; j++)
    {
        u_1[0][j] = mu_2(t, j * hx);
        u_1[My][j] = mu_4(t, j * hx, Ly);
    }


    double** u_2 = malloc((My + 1) * sizeof(double*)); // массив значений u в момент времени t_2
    for(int i = 0; i <= My; i++)
    {
        u_2[i] = malloc((Mx + 1) * sizeof(double));
    }

    while(t < T - tau / 2)
    {
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
        for(int i = 0; i < My + 1; i++)
        {
            u_2[i][0] = mu_1(t);
            u_2[i][Mx] = mu_3(t, Lx, i * hy);
        }

        // самая верхняя строка и самая нижняя строка
        for(int j = 0; j < Mx + 1; j++)
        {
            u_2[0][j] = mu_2(t, j * hx);
            u_2[My][j] = mu_4(t, j * hx, Ly);
        }

        double** tmp = u_0;
        u_0 = u_1;
        u_1 = u_2;
        u_2 = tmp;
    }


    // считаем нормы для того, чтобы сравнить с аналитическим решение
    
    double** u_t = malloc((My + 1) * sizeof(double*)); // массив значений u в момент времени T по точному решению
    for(int i = 0; i <= My; i++)
    {
        u_t[i] = malloc((Mx + 1) * sizeof(double));
    }


    for(int i = 0; i <= My; i++)
    {
        for(int j = 0; j <= Mx; j++)
        {
            u_t[i][j] = f_u_t(T, j * hx, i * hy);
        }
    }

    // считаем норму
    double norm_L_2 = 0, norm_c = 0;
    
    for(int i = 0; i <= My; i++)
    {
        for(int j = 0; j <= Mx; j++)
        {
            norm_L_2 += (u_1[i][j] - u_t[i][j]) * (u_1[i][j] - u_t[i][j]);
            norm_c = fmax(norm_c, fabs(u_1[i][j] - u_t[i][j]));
        }
    }
    norm_L_2 = hx * hy * norm_L_2;
    norm_L_2 = sqrt(norm_L_2);

    //double time2 = omp_get_wtime();

    // освобождаем память
    for(int i = 0; i <= My; i++)
    {
        free(u_0[i]);
        free(u_1[i]);
        free(u_2[i]);
        free(u_t[i]);
    }
    free(u_0);
    free(u_1);
    free(u_2);
    free(u_t);

    printf("%.16lf, %.16lf, Time = %.16lf\n", norm_L_2, norm_c, (double) (clock() - start_time)/CLOCKS_PER_SEC);
}

