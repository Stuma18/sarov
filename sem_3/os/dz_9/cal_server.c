/*
Написать клиент-серверную программу «Калькулятор» :

Процесс-сервер – создает "слушающий" сокет и ждет запросов на соединение от клиентов. 
Приняв запрос, процесс-сервер создает сыновний процесс, который должен обслужить 
клиента и завершиться, а процесс-отец продолжает принимать запросы на соединение 
и создавать новых сыновей. 
Задача сына (обслуживание) – выполнять возможные команды от клиента: 
1) \+ <число> – установить число на которое сервер будет увеличивать числа 
для данного клиента (по умолчанию 1), вернуть клиенту “Ok”; 
2) <число> – запрос на увеличение числа от клиента, задача – увеличить данное число 
на заданное и вернуть клиенту; 
3) \? – получить от сервера число, на которое тот увеличивает числа для текущего клиента; 
4) \- – сообщить серверу о завершении работы, при этом сервер-сын завершается. 

Программа-клиент: 
1) запрашивает у пользователя очередную команду, отправляет ее серверу 
(адрес сервера можно задать в командной строке (передаются в main() через argv), 
или спросить у пользователя первым действием; 
2) получает от сервера ответ и печатает его на экран. 

В качестве программы клиента возможно использование утилит telnet или netcat 
(в разных системах может называться nc, netcat, ncat, pnetcat, не входит в стандарт POSIX). 
*/

// Серверная часть

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>

#define PORT 8080
#define BUFSIZE 1024

void child_process(int sockfd)
{
    int n, num = 1;
    char buffer[BUFSIZE], msg[BUFSIZE];
    
    while (1)
    {
        memset(buffer, 0, BUFSIZE);
        n = read(sockfd, buffer, BUFSIZE);
        buffer[n] = 0;
        
        if (buffer[0] == '+')
        {
            num = atoi(buffer + 1);
            strcpy(msg, "Ok");
        }
        else if (buffer[0] == '?')
        {
            snprintf(msg, BUFSIZE, "%d", num);
        }
        else if (buffer[0] == '-')
        {
            break;
        }
        else
        {
            int client_num = atoi(buffer);
            snprintf(msg, BUFSIZE, "%d", client_num + num);
        }
        
        write(sockfd, msg, strlen(msg));
    }
    
    close(sockfd);
    exit(0);
}

int main()
{
    int sockfd, newsockfd, pid;
    struct sockaddr_in serv_addr, cli_addr;
    socklen_t clilen = sizeof(cli_addr);
    
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(PORT);
    
    bind(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr));
    listen(sockfd, 5);
    
    while (1)
    {
        newsockfd = accept(sockfd, (struct sockaddr *)&cli_addr, &clilen);
        
        if ((pid = fork()) == 0)
        {
            close(sockfd);
            child_process(newsockfd);
        }
        else
        {
            close(newsockfd);
            waitpid(-1, NULL, WNOHANG);
        }
    }
    
    return 0;
}
