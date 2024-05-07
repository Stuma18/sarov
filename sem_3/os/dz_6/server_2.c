/*
С помощью именованного канала организовать клиент-серверную обработку данных следующим образом.
Процесс-сервер запускается первым, обнуляет целую переменную-сумматор,
создает именованный канал, открывает его на чтение, затем в бесконечном цикле 
читает очередное целое из канала (во внутреннем представлении int)
и добавляет считанное число к сумматору. 
Получив сигнал SIGINT (сделать соответствующий обработчик сигнала),
сервер выводит на экран текущее значение сумматора, 
закрывает и удаляет именованный канал, и завершается.
    
Процесс-клиент получает в качестве argv[1] число в строковом виде, 
переводит его во внутреннее представление (число типа int) , 
открывает именованный канал на запись и отправляет число серверу, после чего завершается.

Запустить сразу несколько параллельных клиентов из командной строки можно так: 
$>  client 5 &  client 3  & client 4 & client 1 &
Ответ сервера после нажатия Ctl-C:  sum=13
*/

// Серверный процесс:

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <poll.h>


#define PIPE_NAME "named_pipe"

static int server_sum = 0;

void signal_handler(int signum) {
    printf("Сумма: %d\n", server_sum);
    unlink(PIPE_NAME);
    exit(0);
}

int main() {
    mkfifo(PIPE_NAME, 0666);

    signal(SIGINT, signal_handler);

    int pipe_fd = open(PIPE_NAME, O_RDONLY);

    struct pollfd fds;
    fds.fd = pipe_fd;
    fds.events = POLLIN;

    while (1) {
        int ret = poll(&fds, 1, -1);

        if (ret > 0) {
            int num;
            int read_size = read(pipe_fd, &num, sizeof(int));

            if (read_size > 0) {
                server_sum += num;
            }
        }
    }

    close(pipe_fd);
    return 0;
}

// gcc server_2.c -o server
// ./server