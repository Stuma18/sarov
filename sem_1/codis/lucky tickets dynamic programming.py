import math
from math import fabs

num = int(input())

if num % 2 != 0:
   print('Error: odd number of characters')
   exit()
 

def D_k_n(k, n):
    if n == 0:
        if k == 0:
            return 1
        else:
            return 0
    sum = 0
    for j in range(10):
        sum += D_k_n(k -j, n - 1)
    return sum


def shcat_count(n):
    return D_k_n(9 * num / 2, num)


print(shcat_count(num))
