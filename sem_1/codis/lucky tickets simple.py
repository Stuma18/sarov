n = int(input())
a = 0

if n % 2 != 0:
   print('Error: odd number of characters')
   exit()

for x in range(10 ** (n-1), 10 ** n):

    str_x = str(x)
    num_x = len(str_x)

    str_x1 = str_x[0:num_x // 2]
    str_x2 = str_x[num_x // 2:]
    x1 = int(str_x1)
    x2 = int(str_x2)

    sum1 = 0
    while x1 > 0:
        sum1 = sum1 + x1 % 10
        x1 = x1 // 10

    sum2 = 0
    while x2 > 0:
        sum2 = sum2 + x2 % 10
        x2 = x2 // 10

    if (sum1 == sum2):
        a = a + 1

print(a)
