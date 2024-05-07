/*
Написать программу, моделирующую команду SHELL: 
(здесь pri - имена процессов, argj - аргументы процессов, f.dat - файл входных данных, 
f.res - файл результатов; в каждом из процессов pri использован стандартный ввод-вывод). 

Аргументы, необходимые этой программе, задаются в командной строке. 
pr1; pr2; ... ; prn 
*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>

#define MAX_PROCESS_NAME_LEN 256
#define MAX_ARGUMENTS_LEN 256

void run_process(char *process_name, char *arguments[], char *input_file, char *output_file) {
    int fd_in, fd_out;

    // Проверка наличия файла входных данных и открытие его для чтения
    if (input_file != NULL) {
        fd_in = open(input_file, O_RDONLY);
        if (fd_in < 0) {
            perror("Error opening input file");
            exit(1);
        }

        // Перенаправление стандартного ввода на файл
        if (dup2(fd_in, STDIN_FILENO) == -1) {
            perror("Error redirecting input");
            exit(1);
        }
    }

    // Проверка наличия файла результатов и открытие его для записи
    if (output_file != NULL) {
        fd_out = open(output_file, O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
        if (fd_out < 0) {
            perror("Error opening output file");
            exit(1);
        }

        // Перенаправление стандартного вывода на файл
        if (dup2(fd_out, STDOUT_FILENO) == -1) {
            perror("Error redirecting output");
            exit(1);
        }
    }

    // Запуск процесса
    execvp(process_name, arguments);

    // Вывод сообщения об ошибке, если execvp вернула управление
    perror("Error executing process");
    exit(1);
}

int main(int argc, char *argv[]) {
    char process[MAX_PROCESS_NAME_LEN];
    char *arguments[MAX_ARGUMENTS_LEN];
    char *input_file = NULL;
    char *output_file = NULL;

    int i;
    int process_count = 0;

    for (i = 1; i < argc; i++) {
        // Если встречен разделитель ;
        if (strcmp(argv[i], ";") == 0) {
            // Запуск процесса
            arguments[process_count] = NULL;
            process_count++;

            run_process(process, arguments, input_file, output_file);

            // Сброс переменных
            memset(process, 0, sizeof(process));
            memset(arguments, 0, sizeof(arguments));
            input_file = NULL;
            output_file = NULL;
            process_count = 0;
        } else if (strcmp(argv[i], "f.dat") == 0) {
            // Если встречен файл входных данных
            input_file = argv[i];
        } else if (strcmp(argv[i], "f.res") == 0) {
            // Если встречен файл результатов
            output_file = argv[i];
        } else {
            // Если встречено имя процесса или аргумент
            if (process_count == 0) {
                strcpy(process, argv[i]);
            } else {
                arguments[process_count - 1] = argv[i];
            }
            process_count++;
        }
    }

    // Запуск последнего процесса
    arguments[process_count] = NULL;
    process_count++;

    run_process(process, arguments, input_file, output_file);

    return 0;
}
