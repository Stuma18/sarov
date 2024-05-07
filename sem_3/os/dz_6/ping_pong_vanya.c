
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define MESSAGE_TYPE 1

struct message {
    long type;
    char content[16];
};

static int message_queue_id;

void signal_handler(int signum) {
    msgctl(message_queue_id, IPC_RMID, NULL);
    exit(0);
}

int main() {
    message_queue_id = msgget(IPC_PRIVATE, 0666 | IPC_CREAT);
    signal(SIGINT, signal_handler);

    pid_t pid_brother1, pid_brother2;

    if ((pid_brother1 = fork()) < 0) {
        perror("fork");
        exit(1);
    } else if (pid_brother1 == 0) {
        while (true) {
            struct message received_message;
            msgrcv(message_queue_id, &received_message, sizeof(received_message.content), MESSAGE_TYPE, 0);
            printf("Пинг: %s\n", received_message.content);
            sleep(1);

            struct message sent_message = {MESSAGE_TYPE, "pong"};
            msgsnd(message_queue_id, &sent_message, sizeof(sent_message.content), 0);
        }
    } else {
        if ((pid_brother2 = fork()) < 0) {
            perror("fork");
            exit(1);
        } else if (pid_brother2 == 0) {
            while (true) {
                struct message sent_message = {MESSAGE_TYPE, "ping"};
                msgsnd(message_queue_id, &sent_message, sizeof(sent_message.content), 0);

                struct message received_message;
                msgrcv(message_queue_id, &received_message, sizeof(received_message.content), MESSAGE_TYPE, 0);
                printf("Понг: %s\n", received_message.content);
                sleep(1);
            }
        } else {
            while (wait(NULL) > 0) {
                // Waiting for child processes to terminate
            }
            msgctl(message_queue_id, IPC_RMID, NULL);
        }
    }

    return 0;
}
