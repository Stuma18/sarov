// Клиентская часть

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>

#define PORT 8080
#define BUFSIZE 1024

int main(int argc, char *argv[])
{
    int sockfd;
    struct sockaddr_in serv_addr;
    char buffer[BUFSIZE];
    char ip[INET_ADDRSTRLEN];
    
    if (argc < 2)
    {
        printf("Enter server IP address: ");
        scanf("%s", ip);
    }
    else
    {
        strcpy(ip, argv[1]);
    }
    
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);
    
    inet_pton(AF_INET, ip, &serv_addr.sin_addr);
    
    connect(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr));
    
    while (1)
    {
        printf("Enter command: ");
        scanf("%s", buffer);
        
        write(sockfd, buffer, strlen(buffer));
        
        memset(buffer, 0, BUFSIZE);
        read(sockfd, buffer, BUFSIZE);
        
        printf("Server response: %s\n", buffer);
        
        if (buffer[0] == '-')
        {
            break;
        }
    }
    
    close(sockfd);
    return 0;
}
