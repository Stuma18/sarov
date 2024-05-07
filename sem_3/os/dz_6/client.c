// Клиентский процесс:

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define FIFO_NAME "/tmp/myfifo"

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
    int fd = open(FIFO_NAME, O_WRONLY);
  
    // Отправка числа серверу
    write(fd, &num, sizeof(num));
  
    // Закрытие именованного канала
    close(fd);
  
    return 0;
}