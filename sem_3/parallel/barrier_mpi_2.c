#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <unistd.h>

#define TAG 666

// определяем количество тактов и округляем в большую сторону
int takt(int size)
{
    float x;
    x = log2(size);
    if (x - (int)x != 0)
        return ceil(x);
    else
	    return x;
}

// функция барьера
void barrier(MPI_Comm comm) {
    int rank, size;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int takts = takt(size);     // количество тактов
    
    // этап 1: обмен сообщениями между нитями в порядке возрастания шага начиная (от 1 до takt)
    for (int i = 1; i <= takts; i++)
    {
        int step, offset;
        step = 1 << i;          // шаг: степени 2 (2, 4, 8, 16, 32, 64 и тд)
        offset = step >> 1;     // 1, 2, 4, 8, 16, 32 и тд

        for(int j = 0; j * step < size; j++)
        {
            // определяем отправителя и получателя для каждого потока в зависимости от текущего такта и шага
            int receiver = j * step;
            int sender = (receiver + offset < size) ? (receiver + offset) : (MPI_PROC_NULL);

            if (rank == receiver)
            {
                MPI_Recv(NULL, 0, MPI_CHAR, sender, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            else if (rank == sender)
            {
                MPI_Send(NULL, 0, MPI_CHAR, receiver, TAG, MPI_COMM_WORLD);
            }
        }
    }

    // этап 2: обратный цикл, обмен сообщениями между нитями в порядке убывания шага шага начиная (от takt до 1)
    for (int i = takts; i >= 1; i--)
    {
        int step, offset;
        step = 1 << i;          // ... 64, 32, 16, 8, 4, 2
        offset = step >> 1;     // ... 32, 16, 8, 4, 2, 1

        for(int j = 0; j * step < size; j++)
        {
            // определяем отправителя и получателя для каждого потока в зависимости от текущего такта и шага
            int sender = j * step;
            int receiver = (sender + offset < size) ? (sender + offset) : (MPI_PROC_NULL);

            if (rank == sender)
            {
                MPI_Send(NULL, 0, MPI_CHAR, receiver, TAG, MPI_COMM_WORLD);
            }

            else if (rank == receiver)
            {
                MPI_Recv(NULL, 0, MPI_CHAR, sender, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }
}

int main(int argc, char *argv[]) 
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 1)
    {
		sleep(2);
	}
	else if (rank && !(rank % 3))
	{
		sleep(4);
	}

    printf("Перед барьером, нить %d\n", rank);
    
    // Функция барьера вызова
    barrier(MPI_COMM_WORLD);
    
    printf("После барьера, нить %d\n", rank);

   // barrier(MPI_COMM_WORLD);
    
   // printf("После барьера 2, нить %d\n", rank);
    
    MPI_Finalize();
    return 0;
}
