import math
from math import fabs

n = int(input())

if n % 2 != 0:
    print('Error: odd number of characters')
    exit()

n = n // 2 #divide by two, because in the formula used, n is the length of half of the number


def C_n_k(n, k):
    return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))


sum = 0
for j in range(math.floor(9 * n / 10) + 1):
    sum += ((-1) ** j) * C_n_k(2 * n, j) * C_n_k(11 * n - 10 * j - 1, 9 * n - 10 * j)
    sum = fabs(sum)

print(sum)
