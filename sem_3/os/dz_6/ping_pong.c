/*
Организовать игру Пинг-Понг между двумя процессами-братьями, 
используя синхронизацию с помощью очереди сообщений.
    
Очередь создает (ключ IPC_PRIVATE) и удаляет процесс-отец, сыновья ее наследуют.
Игра идет до нажатия Ctll-C — отец должен обработать эту ситуацию и удалить очередь.
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/msg.h>
#include <sys/types.h>
#include <signal.h>

// Структура сообщения
struct msg_buffer {
    long mtype;
    char mtext[100];
};

static int msgid;

// Функция обработки сигнала SIGINT (Ctrl-C)
void handle_sigint(int sig)
{
    // Удаление очереди сообщений
    msgctl(msgid, IPC_RMID, NULL);
    exit(0);
}

int main() {
    // Создание очереди сообщений
    msgid = msgget(IPC_PRIVATE, 0666 | IPC_CREAT);
    if (msgid == -1) {
        perror("msgget");
        exit(1);
    }

    // Установка обработчика сигнала SIGINT (Ctrl-C)
    signal(SIGINT, handle_sigint);

    // Создание процесса-отца
    pid_t pid = fork();

    if (pid == -1) {
        perror("fork");
        exit(1);
    }

    pid_t pid_2 = fork();

    if (pid_2 == -1) {
        perror("fork");
        exit(1);
    }

    if (pid == 0) {
        // Код процесса-сына, который играет роль "Пинг"
        struct msg_buffer msg;

        while (1) {
            // Чтение сообщения из очереди
            msgrcv(msgid, &msg, sizeof(struct msg_buffer), 1, 0);
            printf("Пинг\n");
            
            // Отправка сообщения в очередь
            msg.mtype = 2;
            msgsnd(msgid, &msg, sizeof(struct msg_buffer), 0);
        }
    } else if (pid_2 == 0) {
        // Код процесса-отца, который играет роль "Понг"
        struct msg_buffer msg;

        while (1) {
            // Отправка сообщения в очередь
            msg.mtype = 1;
            msgsnd(msgid, &msg, sizeof(struct msg_buffer), 0);

            // Чтение сообщения из очереди
            msgrcv(msgid, &msg, sizeof(struct msg_buffer), 2, 0);
            printf("Понг\n");
        }
    } else {
            while (wait(NULL) > 0) {
                // Waiting for child processes to terminate
            }
            msgctl(msgid, IPC_RMID, NULL);
        }

    return 0;
}
