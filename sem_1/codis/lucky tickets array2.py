import time

n = int(input())

if n % 2 != 0:
    print('Error: odd number of characters')
    exit()

n = n // 2

time1 = time.time()

numbers1 = list([1] + [0] * 9 * n)

while n > 0:
    numbers2 = numbers1.copy()
    for i in range(1,10):
        for j in range(len(numbers1)):
            if numbers2[j] == 0:
                break
            numbers1[i + j] += numbers2[j]
    n -= 1

numbers1_sq = [x ** 2 for x in numbers1]
sum = sum(numbers1_sq)

time2 = time.time()
time = time2 - time1

print('amount =', sum)
print('time =', time, 'seconds')
