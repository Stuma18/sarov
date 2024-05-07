/* Пример реализации барьерной синхронизации нитей для разделенной памяти
с использованием семафоров:*/

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>

// Количество нитей
#define NUM_THREADS 4

// Семафоры для синхронизации
sem_t mutex; // Семафор для защиты критической секции
sem_t barrier_sem; // Семафор барьера
int count = 0;

void barrier() 
{
    sem_wait(&mutex); // Блокируем доступ к критической секции
    
    count++;
    if (count == NUM_THREADS) 
    {
        // Последняя нить достигла барьера
        sem_post(&barrier_sem); // Разрешаем ожидающим нитям продолжить
    }
    
    sem_post(&mutex); // Разблокируем доступ к критической секции
    
    sem_wait(&barrier_sem); // Ожидаем, пока все нити достигнут барьера
    count--;
    sem_post(&barrier_sem); // Разрешаем следующей нити продолжить
}

void *thread_func(void *thread_id) 
{
    int id = *(int *)thread_id;
    
    barrier();

    printf("the barrier worked\n");
    
    pthread_exit(NULL);
}

int main() 
{
    pthread_t threads[NUM_THREADS];
    int thread_ids[NUM_THREADS];
    
    // Инициализируем семафоры
    sem_init(&mutex, 0, 1);
    sem_init(&barrier_sem, 0, 0);
    
    // Создаем нити
    for (int i = 0; i < NUM_THREADS; i++) 
    {
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, thread_func, (void *)&thread_ids[i]);
    }
    
    // Ожидаем завершения всех нитей
    for (int i = 0; i < NUM_THREADS; i++) 
    {
        pthread_join(threads[i], NULL);
    }
    
    // Уничтожаем семафоры
    sem_destroy(&mutex);
    sem_destroy(&barrier_sem);
    
    return 0;
}

// gcc barrier_sem.c
// ./a.out
