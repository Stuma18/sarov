/* Пример реализации барьера
с использованием библиотеки MPI и функций MPI_Send и MPI_Recv: */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void barrier(int rank, int size) 
{
    MPI_Status status; 
    
    if (size <= 1) 
    {
        printf("size <= 1\n");
        return;
    }

    // Первый этап
    if (rank == 0) 
    {
        for (int i = 1; i < size; i++) 
        {
            MPI_Send(NULL, 0, MPI_INT, i, 0, MPI_COMM_WORLD); // Отправляем сигнал о вызове барьера
            MPI_Recv(NULL, 0, MPI_INT, i, 0, MPI_COMM_WORLD, &status); // Ждем подтверждение от процесса i
        }
    } 
    else 
    {
        MPI_Recv(NULL, 0, MPI_INT, 0, 0, MPI_COMM_WORLD, &status); // Ждем сигнала о вызове барьера
        MPI_Send(NULL, 0, MPI_INT, 0, 0, MPI_COMM_WORLD); // Отправляем подтверждение о вызове барьера
    }

    // Возвращаем управление после завершения первого этапа
    if (rank == 0)
    {
        for (int i = 1; i < size; i++)
        {
            MPI_Send(NULL, 0, MPI_INT, i, 0, MPI_COMM_WORLD); // Отправляем сигнал об окончании первого этапа
            MPI_Recv(NULL, 0, MPI_INT, i, 0, MPI_COMM_WORLD, &status); // Ждем подтверждение от процесса i
        }
    }
    else
    {
        MPI_Recv(NULL, 0, MPI_INT, 0, 0, MPI_COMM_WORLD, &status); // Ждем сигнала об окончании первого этапа
        MPI_Send(NULL, 0, MPI_INT, 0, 0, MPI_COMM_WORLD); // Отправляем подтверждение об окончании первого этап
    }
}

int main() 
{
    MPI_Init(NULL, NULL);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Получаем ранг текущего процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Получаем количество процессов
    
    if (rank == 0) 
    {
        printf("Number of processes: %d\n", size);
    }
    
    barrier(rank, size);

    printf("Process %d completed barrier\n", rank);

    
    /*
    printf("start work\n");
    barrier(rank, size);
    printf("the barrier worked\n");
    */
    
    
    MPI_Finalize();
    return 0;
}

// mpicc barrier_mpi.c
// mpirun -np 4 ./a.out