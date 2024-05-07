// Клиентский процесс:

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define PIPE_NAME "named_pipe"

int main(int argc, char *argv[]) 
{
    // Проверка наличия аргумента с числом
    if (argc < 2) {
        printf("Необходимо указать число в качестве аргумента\n");
        exit(1);
    }
  
    // Конвертация строки с числом в int
    int num = atoi(argv[1]);
  
    // Открытие именованного канала на запись
    int pipe_fd = open(PIPE_NAME, O_WRONLY);
  
    // Отправка числа серверу
    write(pipe_fd, &num, sizeof(num));
  
    // Закрытие именованного канала
    close(pipe_fd);
  
    return 0;
}

// gcc client_2.c -o client
// ./client 5 & ./client 3 & ./client 4 & ./client 1 