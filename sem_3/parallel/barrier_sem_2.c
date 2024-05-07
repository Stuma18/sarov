#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <semaphore.h>
#include <pthread.h>


// Создаем структуру, в которой лежит rank, size каждого потока и двухмерный массив семафоров sems
struct some_data
{
	int rank, size;
	sem_t** sems;       // Семафоры, двухмерный массив
} typedef parameters;   // Название

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
void barrier(void *data) 
{   
    // Приводим к читабельному виду
    parameters* params = data;
    
    int rank, size;
    rank = params->rank;
	size = params->size;
    
    sem_t** sems = params->sems;    // Из структуры берем массив семафором и присваиваем к sems
    
    int takts = takt(size);     // количество тактов

    // этап 1
    for (int i = 1; i <= takts; i++)
    {
        int step, offset;
        step = 1 << i;          // шаг: степени 2 (2, 4, 8, 16, 32, 64 и тд)
        offset = step >> 1;     // 1, 2, 4, 8, 16, 32 и тд

        for(int j = 0; j * step < size; j++)
        {
            // определяем номера процессов, которые будут синхронизироваться
            int receiver = j * step;
            int sender = receiver + offset;

            if (sender < size)
            {
                if (rank == receiver)
                {
                    sem_post(sems[i - 1] + sender);
                    sem_wait(sems[i - 1] + receiver);
                }

                else if (rank == sender)
                {
                    sem_post(sems[i - 1] + receiver);
                    sem_wait(sems[i - 1] + sender);
                }
            }
        }
    }

    // этап 2: обратный цикл
    for (int i = takts; i >= 1; i--)
    {
        int step, offset;
        step = 1 << i;          // ... 64, 32, 16, 8, 4, 2
        offset = step >> 1;     // ... 32, 16, 8, 4, 2, 1

        for(int j = 0; j * step < size; j++)
        {
            // определяем номера процессов, которые будут синхронизироваться
            int sender = j * step;
            int receiver = sender + offset;

            if (receiver < size)
            {
                if (rank == receiver)
                {
                    sem_post(sems[i - 1] + sender);     // разблокировка
                    sem_wait(sems[i - 1] + receiver);   // блокировка
                }

                else if (rank == sender)
                {
                    sem_post(sems[i - 1] + receiver);   // разблокировка
                    sem_wait(sems[i - 1] + sender);     // блокировка
                }
            }
        }
    }
}

// потоки спят, принт, барьер, принт
void* thread_func(void *data)
{
	parameters* params = data;  // Открываем данные, это нужно чтобы знать rank и size

    int rank, size;
    rank = params->rank,    // Берем rank из данных
	size = params->size;    // Берем size из данных

    if (rank == 1)
    {
		sleep(2);
	}
	else if (rank && !(rank % 3))
	{
		sleep(4);
	}

    printf("Перед барьером, нить %d\n", rank);
    
    barrier(data);
    
    printf("После барьера, нить %d\n", rank);
}


int main(int argc, char **argv) 
{
    if (argc != 2)
	{
		fprintf(stderr, "Надо ввести: %s количество нитей\n", argv[0]);
		return 1;
	}
	
    // количество потоков
	int NUM_THREADS = atoi(argv[1]);

    // считает количество тактов
    int task = takt(NUM_THREADS);

    // Выделяем память для массива потоков 
	pthread_t* threads = malloc(NUM_THREADS * sizeof(pthread_t));   // Создаем динамически массив для нитей

    // Выделяем память для двумерного массива семафоров
	sem_t** sems = malloc(task * sizeof(sem_t*));   // Создаем двумерный динамический массив для семафоров
	for (int i = 0; i < task; ++i)
	{
		sems[i] = malloc(NUM_THREADS * sizeof(sem_t));
		for (int j = 0; j < NUM_THREADS; ++j)
		{
			sem_init(sems[i]+j, 0, 0);
		}
	}

    // Выделяем память для массива параметров
    parameters* params = malloc(NUM_THREADS * sizeof(parameters));  // Создаем динамически массив для параметров (rank, size, sem)

    // Создаем нити
    for (int i = 0; i < NUM_THREADS; i++) 
    {
        params[i].rank = i;
		params[i].size = NUM_THREADS;
		params[i].sems = sems;
        pthread_create(&threads[i], NULL, thread_func, params + i);
    }
    
    // Ожидаем завершения всех нитей
    for (int i = 0; i < NUM_THREADS; i++) 
    {
        pthread_join(threads[i], NULL);
    }
    
    // Отсвобождение памяти и уничтожение семафоров
    free(params);

    for (int i = 0; i < task; ++i)
	{
		for (int j = 0; j < NUM_THREADS; ++j)
		{
			sem_destroy(sems[i]+j);
		}
		free(sems[i]);
	}

    free(sems);

    free(threads);

    return 0;
}
