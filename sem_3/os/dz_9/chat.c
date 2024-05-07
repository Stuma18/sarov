/*
Написать клиент-серверную программу «Чат» 

Процесс-сервер: 
1) создает "слушающий" сокет и ждет запросов на соединение от клиентов; 
2) при поступлении запроса, устанавливается соединение с очередным клиентом, 
от клиента сервер получает имя (ник) вошедшего в "комнату для разговоров", 
клиент заносится в список присутствующих; 
3) всем присутствующим рассылается сообщение, что в комнату вошел такой-то (имя); 
4) от разных клиентов могут поступать реплики – получив реплику от клиента, 
сервер рассылает ее всем "присутствующим" (включая самого автора) с указанием автора реплики; 
5) при разрыве связи (команда \quit) с клиентом сервер сообщает всем, 
что-такой-то (имя) нас покинул (ушел) и выводит его прощальное сообщение. 

Необходима реализация команд: 
а) \users получить от сервера список всех пользователей (имена), которые сейчас онлайн; 
б) \quit <message> – выход из чата с прощальным сообщением. 

В качестве программы клиента возможно использование telnet (или nc, netcat). 
Имена пользователям могут выдаваться автоматически, но они должны быть уникальными. 
Иной подход – сервер запрашивает имя у клиента
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <pthread.h>

#define PORT 8080
#define BUFSIZE 1024
#define MAX_CLIENTS 256

struct client
{
    int sockfd;
    char name[BUFSIZE];
};

struct client clients[MAX_CLIENTS];
pthread_mutex_t clients_mutex = PTHREAD_MUTEX_INITIALIZER;

void broadcast(const char *message, int exclude_sockfd)
{
    pthread_mutex_lock(&clients_mutex);

    for (int i = 0; i < MAX_CLIENTS; i++)
    {
        if (clients[i].sockfd != 0 && clients[i].sockfd != exclude_sockfd)
        {
            write(clients[i].sockfd, message, strlen(message));
        }
    }

    pthread_mutex_unlock(&clients_mutex);
}

void *client_handler(void *arg)
{
    struct client client_info = *(struct client *)arg;
    char buffer[BUFSIZE], message[BUFSIZE * 2];

    snprintf(message, sizeof(message), "%s has entered the chat room.\n", client_info.name);
    broadcast(message, client_info.sockfd);

    while (1)
    {
        memset(buffer, 0, BUFSIZE);
        ssize_t read_bytes = read(client_info.sockfd, buffer, BUFSIZE - 1);
        
        if(read_bytes <= 0)
        {
            break;
        }

        if (strncmp(buffer, "\\users", 6) == 0)
        {
            pthread_mutex_lock(&clients_mutex);

            for (int i = 0; i < MAX_CLIENTS; i++)
            {
                if (clients[i].sockfd != 0)
                {
                    snprintf(message, sizeof(message), "%s\n", clients[i].name);
                    write(client_info.sockfd, message, strlen(message));
                }
            }

            pthread_mutex_unlock(&clients_mutex);
        }
        else if (strncmp(buffer, "\\quit", 5) == 0)
        {
            break;
        }
        else
        {
            snprintf(message, sizeof(message), "%s: %s", client_info.name, buffer);
            broadcast(message, client_info.sockfd);
        }
    }

    snprintf(message, sizeof(message), "%s has left the chat room.\n", client_info.name);
    broadcast(message, client_info.sockfd);

    close(client_info.sockfd);
    pthread_mutex_lock(&clients_mutex);
    clients[client_info.sockfd].sockfd = 0;
    pthread_mutex_unlock(&clients_mutex);

    return NULL;
}

int main()
{
    int sockfd, newsockfd;
    struct sockaddr_in serv_addr, cli_addr;
    socklen_t clilen = sizeof(cli_addr);
    pthread_t thread;

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
        struct client client_info = {
            .sockfd = newsockfd
        };
        read(newsockfd, client_info.name, BUFSIZE - 1);

        pthread_mutex_lock(&clients_mutex);
        clients[newsockfd] = client_info;
        pthread_mutex_unlock(&clients_mutex);

        pthread_create(&thread, NULL, client_handler, &client_info);
    }
    
    return 0;
}

// gcc chat.c -lpthread
// ./a.out 

// nc 127.0.0.1 8080  