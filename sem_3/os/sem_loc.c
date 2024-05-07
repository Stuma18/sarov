#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#define SADDRESS "mysocket"
#define CADDRESS "clientsocket"
#define BUFLEN 40

int main (int argc, char ** argv)
{
    struct sockaddr_un party_addr, own_addr;
    int sockfd;
    int is_server;
    char buf[BUFLEN];
    int party_len;
    int quitting;
    if (argc != 2)
    {
        printf( "Usage: %s client|server.\n", argv[0]);
        return 0;
    }

    quitting = 1;
    is_server =! strcmp (argv[1], "server");
    memset (&own_addr, 0, sizeof(own_addr));
    own_addr.sun_family = AF_UNIX;
    strcpy (own_addr.sun_path, is_server ? SADDRESS: CADDRESS);
    printf("%s\n", own_addr.sun_path);
    if ((sockfd = socket (AF_UNIX, SOCK_DGRAM, 0)) < 0) 
    {
        perror("can't create socket\n");
        return 0;
    }

    // связываем сокет
    unlink (own_addr.sun_path);
    if (bind (sockfd, (struct sockaddr *) &own_addr, sizeof (own_addr.sun_family) + strlen(own_addr.sun_path)+1) < 0)
    {
        perror("can't bind socket!");
        return 0;
    }

    // это — клиент
    if (!is_server) 
    {
        memset(&party_addr, 0, sizeof (party_addr));
        party_addr.sun_family = AF_UNIX;
        strcpy(party_addr.sun_path, SADDRESS);
        printf("type the string:");

        // не пора ли выходить?
        while (gets(buf))
        {
            quitting = (!strcmp(buf, "quit"));
            // считали строку и передаем ее серверу
            if (sendto(sockfd, buf, strlen(buf) + 1, 0, 
                       (struct sockaddr*)& party_addr, sizeof(party_addr.sun_family) + strlen(SADDRESS)+1) != strlen(buf) + 1) 
            {
                perror("client: error writing socket!\n");
                return 0;
            }
            if (recvfrom(sockfd, buf, BUFLEN, 0, NULL, 0) < 0)
            {
                perror("client: error reading socket!\n");
                return 0;
            }
            printf("client: server answered: %s\n", buf);

            if (quitting)
            {
                break;
            }
            perror("type the string:");
        }
        close(sockfd);
        return 0;
    }

    // получаем строку от клиента и выводим на печать
    while (1) 
    {
        party_len = sizeof(party_addr);
        
        if (recvfrom(sockfd, buf, BUFLEN, 0, (struct sockaddr *)&party_addr, &party_len) < 0)
        {
            perror("server: error reading socket!");
            return 0;
        }

        printf("server: received from client: %s\n", buf);

        // не пора ли выходить?
        quitting = (!strcmp(buf, "quit"));

        if (quitting)
            strcpy(buf, "quitting now!");
        else
            if (!strcmp(buf, "ping!"))
                strcpy(buf, "pong!");
            else
                strcpy(buf, "wrong string!");

        // посылаем ответ
        if (sendto(sockfd, buf, strlen(buf) + 1, 0, (struct sockaddr *)&party_addr, party_len) != strlen(buf) + 1) 
        {
            perror("server: error writing socket!\n");
            return 0;
        }

        if (quitting)
        {
            break;
        }
    }

    close(sockfd);
    return 0;
}
