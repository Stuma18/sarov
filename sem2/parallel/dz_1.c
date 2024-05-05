/* 
С помощью технологии MPI реализовать программу для выбора координатора среди 16 процессов, 
находящихся в узлах транспьютерной матрицы размером 4*4, использующую круговой алгоритм.
Все необходимые межпроцессорные взаимодействия реализовать при помощи пересылок MPI типа точка-точка.
Получить временную оценку работы алгоритма. Оценить сколько времени потребуют выборы координатора, 
если время старта равно 100, время передачи байта равно 1 (Ts=100,Tb=1). Процессорные операции, включая 
чтение из памяти и запись в память считаются бесконечно быстрыми.
*/


#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <string.h>


const int CIRCLE_START_RANK = 5;

const int ELECTION_TAG = 1;
const int I_AM_HERE_TAG = 2;
const int COORDINATOR_TAG = 3;

void roll_call(int rank) {
    printf("%d ", rank);
    fflush(stdout);
}

int send_to_next(int rank, int size, int *array, int tag) {
    // пытается отправить сообщение и возвращает ранг следующего активного процесса
    //MPI_Request request;
    //MPI_Status status;
    MPI_Request request[2];
    MPI_Status status;
    int answer;
    int successfully_sent = 0;
    int next = 1;
    while (!successfully_sent) {
        //MPI_Isend(array, size, MPI_INT, (rank + next) % size, tag, MPI_COMM_WORLD, &request);
        //MPI_Irecv(&answer, 1, MPI_INT, (rank + next) % size, I_AM_HERE_TAG, MPI_COMM_WORLD, &request);

        MPI_Isend(array, size, MPI_INT, (rank + next) % size, tag, MPI_COMM_WORLD, &request[0]);
        MPI_Irecv(&answer, 1, MPI_INT, (rank + next) % size, I_AM_HERE_TAG, MPI_COMM_WORLD, &request[1]);

        double start = MPI_Wtime();
        while (!successfully_sent) {
            //MPI_Test(&request, &successfully_sent, &status);

            MPI_Test(&request[0], &successfully_sent, &status);

            // перерыв составляет 1 секунду
            if (MPI_Wtime() - start >= 1) {
                printf("%2d: Не получил подтверждения от %2d.\n", rank, (rank + next) % size);
                fflush(stdout);
                //MPI_Cancel(&request);
                //MPI_Request_free(&request);

                MPI_Cancel(&request[0]);
                MPI_Request_free(&request[0]);

                break;
            }
        }
        next++;
    }
    return status.MPI_SOURCE;
}

int calculate_coordinator(int *array, int size) {
    // запуск процесса с наивысшим рангом координатора
    int i;
    for (i = size - 1; i >= 0; i--) {
        if (array[i]){
            return i;
        }
    }
    return 0;
}

int main(int argc, char* argv[]) {
    setvbuf(stdout, NULL, _IOLBF, BUFSIZ);
    setvbuf(stderr, NULL, _IOLBF, BUFSIZ);
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    srand(time(NULL) * rank);

    bool am_i_alive = true;

    if (rank == CIRCLE_START_RANK) {
        printf("\n\nЖивые процессы:\n");
        fflush(stdout);
    } else {
        am_i_alive = rand() > 0.42 * RAND_MAX;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (am_i_alive) {
        roll_call(rank);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (!am_i_alive) {
        MPI_Finalize();
        return 0;
    }

    int *array = (int *) calloc(size, sizeof(int));

    //MPI_Request request;
    //MPI_Status status;
    
    MPI_Request request[2];
    MPI_Status status;
    int next;

    if (rank == CIRCLE_START_RANK) {
        printf("\nАлгоритм:\n");
        fflush(stdout);
        array[rank] = 1;
        next = send_to_next(rank, size, array, ELECTION_TAG);
    }

    // процесс выбора:
    int answer = 1;
    int new_coordinator;

    //MPI_Irecv(array, size, MPI_INT, MPI_ANY_SOURCE, ELECTION_TAG, MPI_COMM_WORLD, &request);
    //MPI_Wait(&request, &status);

    MPI_Irecv(array, size, MPI_INT, MPI_ANY_SOURCE, ELECTION_TAG, MPI_COMM_WORLD, &request[0]);
    MPI_Wait(&request[2], &status);
    printf("%2d: Получил массив из %2d\n", rank, status.MPI_SOURCE);
    fflush(stdout);

    if (array[rank]) {
        // значение равно 1 -> мы уже были здесь ->
        // нам нужно подвести итоги выборов
        new_coordinator = calculate_coordinator(array, size);
        printf("%2d: Новый координатор: %2d\n", rank, new_coordinator);
        fflush(stdout);
        //MPI_Isend(&answer, 1, MPI_INT, status.MPI_SOURCE, I_AM_HERE_TAG, MPI_COMM_WORLD, &request);
        //MPI_Isend(&new_coordinator, 1, MPI_INT, next, COORDINATOR_TAG, MPI_COMM_WORLD, &request);

        MPI_Isend(&answer, 1, MPI_INT, status.MPI_SOURCE, I_AM_HERE_TAG, MPI_COMM_WORLD, &request[0]);
        MPI_Isend(&new_coordinator, 1, MPI_INT, next, COORDINATOR_TAG, MPI_COMM_WORLD, &request[0]);
    } else {
        // обновление массива и отправка следующему
        array[rank] = 1;
        //MPI_Isend(&answer, 1, MPI_INT, status.MPI_SOURCE, I_AM_HERE_TAG, MPI_COMM_WORLD, &request);

        MPI_Isend(&answer, 1, MPI_INT, status.MPI_SOURCE, I_AM_HERE_TAG, MPI_COMM_WORLD, &request[1]);
        next = send_to_next(rank, size, array, ELECTION_TAG);
    }

    // второй круг: отправка нового координатора
    //MPI_Irecv(&new_coordinator, 1, MPI_INT, MPI_ANY_SOURCE, COORDINATOR_TAG, MPI_COMM_WORLD, &request);
    //MPI_Wait(&request, &status);

    MPI_Irecv(&new_coordinator, 1, MPI_INT, MPI_ANY_SOURCE, COORDINATOR_TAG, MPI_COMM_WORLD, &request[1]);
    MPI_Wait(&request[2], &status);

    if (rank != CIRCLE_START_RANK) {
        printf("%2d: Новый координатор: %2d\n", rank, new_coordinator);
        fflush(stdout);
        //MPI_Isend(&new_coordinator, 1, MPI_INT, next, COORDINATOR_TAG, MPI_COMM_WORLD, &request);

        MPI_Isend(&new_coordinator, 1, MPI_INT, next, COORDINATOR_TAG, MPI_COMM_WORLD, &request[1]);
    }
    
    //MPI_Wait(&request, &status);
    //MPI_Waitall(1, &request, &status);

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}

// mpicc dz_1.c -o main
// mpirun --oversubscribe -n 36 main
