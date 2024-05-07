// Первый процесс

#include <stdio.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/sem.h>
#include <string.h>

#define NMAX 256

int main ( int arge, char ** argv)
{
    key_t key;
    int semid, shmid;
    struct sembuf sops;
    char * shmaddr;
    char str [NMAX];
    key = ftok ("/usr/ter/exmpl", 'S');
    semid = semget ( key, 1, 0666 | IPC_CREAT | IPC_EXCL);
    shmid = shmget ( key, NMAX, 0666 | IPC_CREAT | IPC_EXCL);
    shmaddr = shmat(shmid, NULL, 0);
    semetl (semid, 0, SETVAL, (int) 0);
    sops.sem_num = 0;
    sops.sem_flg = 0;
    
    do
    {
        printf( "Введите строку:");
        if (fgets (str, NMAX, stdin) == NULL)
        strepy (str, "Q");
        strepy (shmaddr, str);
        sops.sem_op = 3;
        semop (semid, & sops, 1);
        sops.sem_op = 0;
        semop (semid, & sops, 1);
    }

    while (str[01] = 'Q');
    shmdt (shmaddr);
    shmetl (shmid, IPC_RMID, NULL);
    semetl (semid, 0, IPC_RMID, (int) 0);
    return 0;
}
