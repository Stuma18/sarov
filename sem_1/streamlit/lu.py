import numpy as np
def decLU(A):
  #Returns the decomposition LU for matrix A
    n = len (A)
    LU = np.copy(A)
    for j in range(0, n - 1):
        for i in range(j + 1, n):
            if LU[i, j] != 0:
                u = LU[i, j] / LU[j ,j]
                LU[i, j + 1:n] = LU[i, j + 1:n] - u * LU[j, j + 1:n]
                LU[i, j] = u
    return LU
  
def solveLU(A, f):
  #Solve the linear system Ax = b
    n = len (A)
  #LU decomposition
    LU = decLU(A)
    x = np.copy(f)
  #forward substitution
    for i in range(1, n):
        x[i] = x[i] - np.dot(LU[i, 0:i], x[0:i])
  #back substitution process
    for i in range(n-1, -1, -1):
        x[i] = (x[i] - np.dot(LU[i, i+1:n], x[i+1:n])) / LU[i, i]
    return x
