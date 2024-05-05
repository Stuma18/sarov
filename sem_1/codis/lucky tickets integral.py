import scipy
import numpy
from scipy import integrate

n = int(input())

if n % 2 != 0:
    print('Error: odd number of characters')
    exit()

def f(x):
    return ((numpy.sin(10 * x)) / (numpy.sin(x))) ** n

#res = [int(x) for x in str(v, err = integrate.quad(f, 0, scipy.pi))]
v, err = integrate.quad(f, 0, scipy.pi)

#sum = round(1 / scipy.pi * v)

#x=list(filter(lambda i:(i),str(sum = round(1 / scipy.pi * v))))

sum = round(1 / scipy.pi * v)

#res = [int(x) for x in str(sum)]

print(sum)
print(v)
#print(res)