#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int ch;

int E();
int M();

int Op()
{
    int x, y, z;
    if (ch =='(')
    { 
        ch = getchar(); 
        x = E();
        if (ch ==')') 
        {
            ch = getchar(); 
            return x;
        }
        else 
        {
            printf ("Ошибка: не хватает )");
            exit (1);
        }
    }
    else if (ch == '-')
    { 
        ch = getchar();
        printf("0 ");
        x = Op();
        printf("- ");
        return -x;
    }
    else if (ch >= '0' && ch <= '9')
    {
        z = ch - '0';
        printf("%d ", z);
        ch = getchar();
        return z;
    }
    else 
    {
        printf ("Ошибка: неправильный операнд )");
        exit (1);
    }
}

int M()
{
    int x, y, z;
    x = Op();
    while (ch =='^')
    {
        z=ch;
        ch = getchar();
        y = Op();
        printf("^ ");
        x = pow(x, y);
    }
    return x;
}

int T()
{
    int x, y, z;
    x = M();
    while (ch =='*' || ch == '/')
    {
        z = ch;
        ch = getchar();
        y = M();
        if (z == '*') 
        {
            printf("* ");
            x *= y;
        }
        else
        {
            printf("/ ");
            x /= y;
        }
    }
    return x;
}


int E()
{
    int x, y, z;
    x = T();
    while (ch =='+' || ch == '-')
    {
        z = ch;
        ch = getchar();
        y = T();
        if (z =='+') 
        {
            printf("+ ");
            x += y; 
        }
        else 
        {
            printf("- ");
            x -= y;
        }
    }
    return x;
}


int main()
{ 
    int k; 
    ch = getchar();
    k = E();
    if (ch != '\n')
        { printf(" End of expression expected\n");
        return 1;
        }
    printf("\n Result = %d\n",k);
    return 0;
}

