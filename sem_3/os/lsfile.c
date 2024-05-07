#include <sys/types.h>
#include <sys/stat.h>
#include <pwd.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/sysmacros.h>
#include <string.h>
#include <dirent.h>

#define MAX_PATH 1024

void dirwalk(char* dir, void (*fcn)(char*));
 
// показать тип файла в первой позиции выходной строки 
void display_file_type ( int st_mode ) 
{                                   
    switch ( st_mode & S_IFMT )
    {
        case S_IFDIR:  putchar ( 'd' ); return;
        case S_IFCHR:  putchar ( 'c' ); return;
        case S_IFBLK:  putchar ( 'b' ); return;
        case S_IFREG:  putchar ( '-' ); return;
        case S_IFLNK:  putchar ( 'l' ); return;
        case S_IFSOCK: putchar ( 's' ); return;
    }
} 
 
// показать права доступа для владельца, группы и прочих пользователей, а также все спец.флаги 
void display_permission ( int st_mode )
{
  static const char xtbl[10] = "rwxrwxrwx";
  char     amode[10];
  int      i, j;
 
  for ( i = 0, j = ( 1 << 8 ); i < 9; i++, j >>= 1 )
    amode[i] = ( st_mode&j ) ? xtbl[i]: '-';
  if ( st_mode & S_ISUID )   amode[2]= 's';
  if ( st_mode & S_ISGID )   amode[5]= 's';
  if ( st_mode & S_ISVTX )   amode[8]= 't';
  amode[9]='\0';
  printf ( "%s ",amode );
}
 
// перечислить атрибуты одного файла
void long_list ( char * path_name )
{
  struct stat     statv;
  struct passwd  *pw_d;
 
  if ( lstat ( path_name, &statv ) )
  { 
    perror ( path_name ); 
    return;
  }
  display_file_type ( statv.st_mode );
  display_permission ( statv.st_mode );
  printf ( "%ld ",statv.st_nlink );  // значение счетчика жестких связей
  pw_d = getpwuid ( statv.st_uid ); // преобразовать UID в имя пользователя
  printf ( "%s ",pw_d->pw_name );   // и напечатать его
 
  if (
      ( statv.st_mode & S_IFMT) == S_IFCHR  ||
      ( statv.st_mode & S_IFMT) == S_IFBLK
     )
    // показать старший и младший номера устройства
    printf ( "%d, %d", major(statv.st_rdev), minor(statv.st_rdev) );
  else
    // или размер файла
    printf ( "%ld", statv.st_size );
  //  показать имя файла
  printf ( "     %s\n", path_name );

  if ((statv.st_mode & S_IFMT) == S_IFDIR)
  {
    dirwalk(path_name, long_list);
  }
}

void dirwalk(char* dir, void (*fcn)(char*))
{
    char name[MAX_PATH];
    struct dirent* dp;
    DIR* dfd;

    if ((dfd = opendir(dir)) == NULL)
    {
        fprintf(stderr, "dirwalk: cannot open file\n");
        return;
    }

    while ((dp = readdir(dfd)) != NULL)
    {
        if (strcmp(dp->d_name, ".") == 0 || strcmp(dp->d_name, "..") == 0)
            continue;
        
        if (strlen(dir) + strlen(dp->d_name) + 2 > sizeof(name))
            fprintf(stderr, "dirwalk: too long name\n");

        else
        {
            sprintf(name, "%s/%s", dir, dp->d_name);
            (*fcn)(name);
        }
    }
    closedir(dfd);
}
 
// главный цикл отображения атрибутов для каждого файла
int main ( int argc, char * argv[] )
{
  if ( argc == 1 )
  {
    fprintf ( stderr, "usage: %s <path name> ...\n", argv[0] );
    dirwalk(".", long_list);
  }
  else
    while ( argc-- != 1 ) 
      dirwalk(*++argv, long_list);
  return 0;
}