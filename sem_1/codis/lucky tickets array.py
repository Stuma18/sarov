import time

n = int(input())

if n % 2 != 0:
    print('Error: odd number of characters')
    exit()

time1 = time.time() 

massif1 = [1] + 9 * n * [0] 
while n > 0: 
    massif2 = massif1.copy() 
    for i in range(1,10): 
        for j in range(0, len(massif1)): 
            if massif2[j] == 0: 
                break 
            massif1[i + j] = massif1[i + j] + massif2[j] 
    n = n - 1 
massif1_sq = [x ** 2 for x in massif1] 
sum = sum(massif1_sq)

time2 = time.time()
time = time2 - time1 

print('amount = ', sum)
print('time = ', time, 'seconds')
