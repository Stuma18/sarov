/*
Организовать игру Пинг-Понг между двумя процессами через семафоры и разделяемую память.
Первый процесс создает массив из двух семафоров и устанавливает значения <1, 0> для него,  
записывает в разделяемую память слово «Ping».
Далее первый процесс в бесконечном цикле : опускает первый семафор, 
cчитывает из разделяемой памяти слово и печатает на экран, спит одну секунду — sleep(1), 
записывает в разделяемую память слово Pong,
после чего поднимает второй семафор и идет на начало цикла.
Второй процесс подсоединяется к созданным ресурсам и  в бесконечном цикле: 
опускает второй семафор, cчитывает из разделяемой памяти слово и печатает на экран, 
спит одну секунду — sleep(1), 
записывает в разделяемую память слово Ping,
после чего поднимает первый семафор и идет на начало цикла.
При получении сигнала SIGINT первый процесс удаляет массив семафоров и 
разделяемую память и завершается, а второй просто завершается.
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/sem.h>

#define SHM_SIZE 1024

// Создаем глобальные переменные для идентификаторов семафоров и разделяемой памяти
int sem_id;
int shm_id;
char *shm_ptr;

// Обработчик сигнала SIGINT
void sigint_handler(int signum) {
    // Удаляем семафоры и разделяемую память
    semctl(sem_id, 0, IPC_RMID);
    shmctl(shm_id, IPC_RMID, NULL);
    exit(0);
}

int main() {
    // Создаем ключ для семафоров и разделяемой памяти
    key_t key = ftok(".", 'P');

    // Создаем массив из двух семафоров и устанавливаем значения <1, 0>
    sem_id = semget(key, 2, IPC_CREAT | 0644);
    semctl(sem_id, 0, SETVAL, 1);
    semctl(sem_id, 1, SETVAL, 0);

    // Создаем разделяемую память и записываем слово "Ping"
    shm_id = shmget(key, SHM_SIZE, IPC_CREAT | 0644);
    shm_ptr = shmat(shm_id, NULL, 0);
    sprintf(shm_ptr, "Ping");

    // Подключаем обработчик сигнала SIGINT
    signal(SIGINT, sigint_handler);

    // Создаем второй процесс
    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        exit(1);
    } else if (pid == 0) {
        // Код второго процесса
        while (1) {
            // Опускаем второй семафор
            struct sembuf sem_op;
            sem_op.sem_num = 1;
            sem_op.sem_op = -1;
            sem_op.sem_flg = 0;
            semop(sem_id, &sem_op, 1);

            // Считываем из разделяемой памяти слово и печатаем на экран
            printf("%s\n", shm_ptr);

            // Спим одну секунду
            sleep(1);

            // Записываем в разделяемую память слово "Ping"
            sprintf(shm_ptr, "Ping");

            // Поднимаем первый семафор
            sem_op.sem_num = 0;
            sem_op.sem_op = 1;
            semop(sem_id, &sem_op, 1);
        }
    } else {
        // Код первого процесса
        while (1) {
            // Опускаем первый семафор
            struct sembuf sem_op;
            sem_op.sem_num = 0;
            sem_op.sem_op = -1;
            sem_op.sem_flg = 0;
            semop(sem_id, &sem_op, 1);

            // Считываем из разделяемой памяти слово и печатаем на экран
            printf("%s\n", shm_ptr);

            // Спим одну секунду
            sleep(1);

            // Записываем в разделяемую память слово "Pong"
            sprintf(shm_ptr, "Pong");

            // Поднимаем второй семафор
            sem_op.sem_num = 1;
            sem_op.sem_op = 1;
            semop(sem_id, &sem_op, 1);
        }
    }

    return 0;
}

