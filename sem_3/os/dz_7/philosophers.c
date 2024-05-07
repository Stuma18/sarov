/*
Реализовать программу, которая запускает пять процессов-философов 
из задачи о пяти философах (при этом каждый философ печатает на экран свой номер (имя), 
и что сейчас делает (гуляет, кушает)).

Код создает 5 потоков, каждый из которых представляет философа. 
Философы поочередно гуляют и кушают, соблюдая правило, 
что философ должен сначала взять вилки слева и справа, а затем положить их обратно после еды. 
Чтобы избежать дедлока, используются семафоры для вилок и семафор num для ограничения количества философов, 
которые одновременно подсаживаются за стол.
*/


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <semaphore.h>

#define N 5 // Количество философов

sem_t forks[N]; // Семафоры для вилок
sem_t num; // Семафор для ограничения количества философов

void *philosopher(void *arg) {
    int *id_ptr = (int *)arg;
    int id = *id_ptr;

    while (1) {
        printf("Философ %d гуляет\n", id);

        // Захват семафора num
        sem_wait(&num);

        // Попытка взять вилку слева
        sem_wait(&forks[id]);

        // Попытка взять вилку справа
        sem_wait(&forks[(id + 1) % N]);

        printf("Философ %d кушает\n", id);

        // Освобождение вилок
        sem_post(&forks[id]);
        sem_post(&forks[(id + 1) % N]);

        // Освобождение семафора num
        sem_post(&num);

        sleep(rand() % 5); // Сон

        printf("Философ %d сыт, возвращается гулять\n", id);
    }

    pthread_exit(NULL);
}

int main() {
    pthread_t philosophers[N];
    int ids[N];

    // Инициализация семафоров
    for (int i = 0; i < N; i++) {
        sem_init(&forks[i], 0, 1); // Вилки изначально доступны
    }

    sem_init(&num, 0, 4); // Изначально 4 философа могут сесть за стол

    // Создание потоков для философов
    for (int i = 0; i < N; i++) {
        ids[i] = i;
        pthread_create(&philosophers[i], NULL, philosopher, &ids[i]);
    }

    // Ожидание завершения потоков
    for (int i = 0; i < N; i++) {
        pthread_join(philosophers[i], NULL);
    }

    // Уничтожение семафоров
    for (int i = 0; i < N; i++) {
        sem_destroy(&forks[i]);
    }

    sem_destroy(&num);

    return 0;
}
