#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h> 


int main (int argc, char ** argv)   /* PID=2021 */
{
   if (fork()==0) { /*PID = 2022 */
      printf("%d %d\n", getppid(), getpid()); 
      return 0;
   }
   if (fork()==0) { /*PID = 2023 */
      printf("%d\n", getpid());
      return 0;
   }
   return 0;
}