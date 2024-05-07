#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>

double phi(double x)
{
    return sin(3 * x);
}

double psi(double x)
{
    return - 2 * sin(3 * x);
}

double f(double a, double t, double x)
{
    return (9 * a * a - 4) * cos(2 * t) * sin(3 * x);
}

double mu_1(double t)
{
    return 0;
}

double mu_2(double t, double L)
{
    return cos(2 * t) * sin(3 * L);
}

double f_u_t(double t, double x)
{
    return cos(2 * t) * sin(3 * x);
}

int main() {
    //double time1 = omp_get_wtime();

    clock_t start_time = clock();

    int N, M;
    N = 200; // размер сетки по t
    M = 100; // размер сетки по пространству

    double T, L;
    T = 1; // сетка по времени от [0,T]
    L = 1; // сетка по пространству [0,L]

    double tau, h;
    tau = T / N; // шаг по времени
    h = L / M; // шаг по пространству

    double a = 1;

    double* u_0 = malloc((M + 1) * sizeof(double)); // массив значений u в момент времени 0

    for(int i = 0; i < M + 1; i++)
    {
        u_0[i] = phi(i * h);
    }

    double* u_1 = malloc((M + 1) * sizeof(double)); // массив значений u в момент времени t_1

    double t = 0; // текущее время

    for(int i = 1; i < M; i++) // все кроме граничных условий
    {
        u_1[i] = u_0[i] + tau * (psi(i * h) + tau / 2 * (a * a) * ((u_0[i+1] - 2 * u_0[i] + u_0 [i - 1]) / (h * h) + f(a, t, i * h)));
    }

    t += tau;

    // самая левая ячейка
    u_1[0] = mu_1(t);

    // самая правая ячейка
    u_1[M] = mu_2(t, L);

    double* u_2 = malloc((M + 1) * sizeof(double)); // массив значений u в момент времени t_2

    while(t < T - tau / 2)
    {
        for(int i = 1; i < M; i++)
        {
            u_2[i] = 2 * u_1[i] - u_0[i] + tau * tau * (a * a * ((u_1[i + 1] - 2 * u_1[i] + u_1[i - 1]) / (h * h)) + f(a, t, i * h));
        }

        t += tau;

        // самая левая ячейка
        u_2[0] = mu_1(t);

        // самая правая ячейка
        u_2[M] = mu_2(t, L);

        double* tmp = u_0;
        u_0 = u_1;
        u_1 = u_2;
        u_2 = tmp;
    }


    // считаем нормы для того, чтобы сравнить с аналитическим решение
    
    double* u_t = malloc((M + 1) * sizeof(double)); // массив значений u в момент времени T по точному решению

    for(int i = 0; i <= M; i++)
    {
        u_t[i] = f_u_t(T, i * h);
    }

    // считаем норму
    double norm_L_2 = 0, norm_c = 0;
    
    for(int i = 0; i <= M; i++)
    {
        norm_L_2 += (u_2[i] - u_t[i]) * (u_2[i] - u_t[i]);
        norm_c = fmax(norm_c, fabs(u_2[i] - u_t[i]));
    }
    norm_L_2 = h * norm_L_2;
    norm_L_2 = sqrt(norm_L_2);

    //double time2 = omp_get_wtime();

    printf("%.16lf, %.16lf, Time = %.16lf\n", norm_L_2, norm_c, (double) (clock() - start_time)/CLOCKS_PER_SEC);
}

