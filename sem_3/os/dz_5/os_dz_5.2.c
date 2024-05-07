/*
Написать программу, моделирующую команду SHELL: 
(здесь pri - имена процессов, argj - аргументы процессов, f.dat - файл входных данных, 
f.res - файл результатов; в каждом из процессов pri использован стандартный ввод-вывод). 

Аргументы, необходимые этой программе, задаются в командной строке. 
pr1 | pr2 > f.res 
*/


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>

int main(int argc, char *argv[]) {
    // Проверяем, что количество аргументов не меньше 5 (название программы, pr1, pr2, >, f.res)
    if (argc < 5) {
        printf("Недостаточно аргументов.\n");
        return 1;
    }

    // Создаем pipe для связи между процессами pr1 и pr2
    int pipefd[2];
    if (pipe(pipefd) == -1) {
        perror("pipe");
        return 1;
    }

    // Создаем дочерний процесс pr1
    pid_t pr1_pid = fork();
    if (pr1_pid == -1) {
        perror("fork");
        return 1;
    }

    if (pr1_pid == 0) { // Код для дочернего процесса pr1
        // Закрываем писательскую сторону pipe
        close(pipefd[0]);

        // Перенаправляем stdout на писательскую сторону pipe
        dup2(pipefd[1], STDOUT_FILENO);

        // Запускаем прогрмму pr1 с переданными аргументами
        execvp(argv[1], argv + 1);

        // Если execvp вернулся, значит произошла ошибка
        perror("execvp");
        return 1;
    } else { // Код для родительского процесса
        // Создаем дочерний процесс pr2
        pid_t pr2_pid = fork();
        if (pr2_pid == -1) {
            perror("fork");
            return 1;
        }

        if (pr2_pid == 0) { // Код для дочернего процесса pr2
            // Закрываем читательскую сторону pipe
            close(pipefd[1]);

            // Перенаправляем stdin на читательскую сторону pipe
            dup2(pipefd[0], STDIN_FILENO);

            // Запускаем программу pr2 с переданными аргументами
            execvp(argv[2], argv + 2);

            // Если execvp вернулся, значит произошла ошибка
            perror("execvp");
            return 1;
        } else { // Код для родительского процесса
            // Закрываем оба конца pipe, так как они уже не нужны родительскому процессу
            close(pipefd[0]);
            close(pipefd[1]);

            // Ожидаем завершения выполнения процесса pr2
            int status2;
            waitpid(pr2_pid, &status2, 0);
            
            // Проверяем, что процесс pr2 успешно завершился
            if (WIFEXITED(status2) && WEXITSTATUS(status2) == 0) {
                // Открываем файл f.res для записи
                int fd_res = open(argv[4], O_WRONLY | O_CREAT | O_TRUNC, 0666);
                if (fd_res == -1) {
                    perror("open");
                    return 1;
                }

                // Перенаправляем stdout на файл f.res
                dup2(fd_res, STDOUT_FILENO);

                // Запускаем программу pr1 с переданными аргументами вновь (потому что первая программа pr1 уже была запущена в дочернем процессе)
                execvp(argv[1], argv + 1);

                // Если execvp вернулся, значит произошла ошибка
                perror("execvp");
                return 1;
            } else {
                printf("Процесс pr2 завершился с ошибкой.\n");
                return 1;
            }
        }
    }

    return 0;
}