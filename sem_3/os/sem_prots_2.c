// Второй процесс

#include <stdio.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/sem.h>
#include <string.h>

#define NMAX 256

int main (int arge, char ** argy)
{
    key_t key;
    int semid, shmid; 
    struct sembuf sops;
    char * shmaddr;
    char str [NMAX];
    key = ftok ("/ust/ter/exmpl", 'S');
    semid = semget (key, 1, 0666);
    shmid = shmget (key, NMAX, 0666);
    shmaddr = shmat (shmid, NULL, 0);
    sops.sem_num = 0;
    sops.sem_flg = 0;
    
    do
    {
        printf ( "Waiting... \n");
        sops.sem_op = -2 ;
        semop (semid, &sops, 1);
        strepy (str, shmaddr);
        if (str[0] == 'Q') 
            shmdt (shmaddr);
        sops.sem_op = -1;
        semop (semid, &sops, 1) ;
        printf ("Read from shared memory: %s\n", str);
    }
    while (str[0] != 'Q');
    
    return 0;
}