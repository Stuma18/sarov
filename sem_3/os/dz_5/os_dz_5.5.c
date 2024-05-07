/*
Написать программу, моделирующую команду SHELL pr1&&pr2 
(выполнить pr1; в случае успешного завершения pr1 выполнить pr2, иначе за-вершить работу). 
Имена процессов задаются в командной строке. 
Процесс считается выполненным успешно, если он вернул 0.
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: ./program pr1 pr2\n");
        return 1;
    }

    int pr1_status;
    pid_t pr1_pid = fork();
    if (pr1_pid == 0) 
    {
        // Child process for pr1
        execlp(argv[1], argv[1], NULL);
        // execlp() only returns if there was an error
        fprintf(stderr, "Failed to execute pr1\n");
        exit(1);
    } 
    else if (pr1_pid < 0) 
    {
        fprintf(stderr, "Failed to fork pr1\n");
        return 1;
    } 
    else 
    {
        // Parent process
        waitpid(pr1_pid, &pr1_status, 0);
        if (WIFEXITED(pr1_status) && WEXITSTATUS(pr1_status) == 0) {
            // pr1 completed successfully
            pid_t pr2_pid = fork();
            if (pr2_pid == 0) 
            {
                // Child process for pr2
                execlp(argv[2], argv[2], NULL);
                // execlp() only returns if there was an error
                fprintf(stderr, "Failed to execute pr2\n");
                exit(1);
            } 
            else if (pr2_pid < 0) 
            {
                fprintf(stderr, "Failed to fork pr2\n");
                return 1;
            } 
            else 
            {
                // Parent process
                waitpid(pr2_pid, NULL, 0);
            }
        }
    }

    return 0;
}
