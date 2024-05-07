/*
Написать программу, моделирующую команду SHELL: 
(здесь pri - имена процессов, argj - аргументы процессов, f.dat - файл входных данных, 
f.res - файл результатов; в каждом из процессов pri использован стандартный ввод-вывод). 

Аргументы, необходимые этой программе, задаются в командной строке. 
pr1 < f.dat > f.res 
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

#define MAX_ARGS 10  // Максимальное количество аргументов процесса

void execute_process(char *process_name, char *args[], char *input_file, char *output_file) {
    pid_t pid;
    int status;

    pid = fork();  // Создание нового процесса

    if (pid < 0) {
        perror("Не удалось создать процесс");
        exit(EXIT_FAILURE);
    } else if (pid == 0) {
        // Дочерний процесс

        // Перенаправление стандартного ввода
        if (input_file != NULL) {
            freopen(input_file, "r", stdin);
        }

        // Перенаправление стандартного вывода
        if (output_file != NULL) {
            freopen(output_file, "w", stdout);
        }

        // Запуск процесса
        execvp(process_name, args);

        perror("Ошибка при запуске процесса");
        exit(EXIT_FAILURE);
    } else {
        // Родительский процесс
        wait(&status);  // Ожидание завершения дочернего процесса
    }
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Программа должна быть запущена с аргументами: pr1 < f.dat > f.res\n");
        return EXIT_FAILURE;
    }

    char *process_name = argv[1];  // Имя процесса
    char *input_file = argv[2];  // Файл входных данных
    char *output_file = argv[3];  // Файл результатов

    // Аргументы процесса
    char *args[MAX_ARGS];
    int num_args = argc - 4;

    if (num_args > MAX_ARGS) {
        printf("Превышено максимальное количество аргументов\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < num_args; i++) {
        args[i] = argv[i + 4];
    }
    args[num_args] = NULL;

    execute_process(process_name, args, input_file, output_file);

    return EXIT_SUCCESS;
}
