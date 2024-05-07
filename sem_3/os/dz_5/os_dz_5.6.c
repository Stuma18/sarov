/*
Организовать игру в волейбол  (по аналогии с игрой  в Пинг-Понг в материалах) 
между тремя китайскими министрами по имени  Пинг, Панг и Понг. 
Министры-процессы порождаются по схеме: 
 
                [ ]
               /    \
             /        \
        Pang       Ping
                          \
                            \
                             Pong
 
Передача "мяча"  осуществляется через каналы (3 канала) по кругу: 
либо по часовой стрелке Pang-->Ping-->Pong-->Pang-->...  либо против часовой.
*/

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/wait.h>

int main() {
    // Создаем 3 канала для обмена мячом
    int channels[3][2];
    for (int i = 0; i < 3; i++) {
        if (pipe(channels[i]) == -1) {
            perror("Ошибка при создании канала");
            exit(1);
        }
    }

    // Создаем процессы для министров
    pid_t pang_pid, ping_pid, pong_pid;
    pang_pid = fork();

    if (pang_pid == -1) {
        perror("Ошибка при создании процесса Pang");
        exit(1);
    } else if (pang_pid == 0) {
        // Код процесса Pang
        close(channels[0][1]); // Закрываем конец записи для канала Pang->Ping
        close(channels[2][0]); // Закрываем конец чтения для канала Pong->Pang
        
        while(1) {
            int ball;
            
            // Получаем мяч от Ping
            read(channels[1][0], &ball, sizeof(ball));
            printf("Pang получил мяч\n");
            
            // Передаем мяч Pong
            write(channels[2][1], &ball, sizeof(ball));
        }
    } else {
        ping_pid = fork();

        if (ping_pid == -1) {
            perror("Ошибка при создании процесса Ping");
            exit(1);
        } else if (ping_pid == 0) {
            // Код процесса Ping
            close(channels[1][1]); // Закрываем конец записи для канала Ping->Pong
            close(channels[0][0]); // Закрываем конец чтения для канала Pang->Ping
            
            while(1) {
                int ball;
                
                // Получаем мяч от Pong
                read(channels[2][0], &ball, sizeof(ball));
                printf("Ping получил мяч\n");
                
                // Передаем мяч Pang
                write(channels[0][1], &ball, sizeof(ball));
            }
        } else {
            pong_pid = fork();

            if (pong_pid == -1) {
                perror("Ошибка при создании процесса Pong");
                exit(1);
            } else if (pong_pid == 0) {
                // Код процесса Pong
                close(channels[2][1]); // Закрываем конец записи для канала Pong->Pang
                close(channels[1][0]); // Закрываем конец чтения для канала Ping->Pong
                
                while(1) {
                    int ball;
                    
                    // Получаем мяч от Pang
                    read(channels[0][0], &ball, sizeof(ball));
                    printf("Pong получил мяч\n");
                    
                    // Передаем мяч Ping
                    write(channels[1][1], &ball, sizeof(ball));
                }
            } else {
                // Код родительского процесса
                close(channels[0][0]); // Закрываем конец чтения для канала Pang->Ping
                close(channels[1][0]); // Закрываем конец чтения для канала Ping->Pong
                close(channels[2][0]); // Закрываем конец чтения для канала Pong->Pang
                
                // Передаем мяч Pang
                int ball = 1;
                write(channels[0][1], &ball, sizeof(ball));
                
                // Ожидаем завершения игры
                wait(NULL);
            }
        }
    }

    return 0;
}
