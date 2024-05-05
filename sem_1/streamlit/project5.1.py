import streamlit as st


chart_visual = st.sidebar.radio('Содержание', 
    ('Главная', 'Постановка задачи', 'Матрица', 'Описание алгоритма', 'Код с NumPy', 'Код со SciPy', 'Параметрические расчеты со SciPy', 'Сравнение'))

if chart_visual == 'Главная':
  st.header("Прямые методы линейной алгебры")
  st.subheader("Задание 5.1")
  st.subheader("Подготовила студентка первого курса магистратуры, группы СТФИ-122")
  st.header("Студеникина Мария")

if chart_visual == 'Постановка задачи':
  st.header('Постановка задачи')
  st.write(r"""
Напишите программу, реализующую решение системы линейных алгебраических
уравнений на основе $LU$- разложения. 

С ее помощью найдите решение системы

$\begin{aligned} 
  A x = f
\end{aligned}$

с матрицей

$\begin{aligned} 
  a_{ij} = \left \{
    \begin{array}{rl}
        1, &  i=j, \\
        -1,&  i < j, \\
        0, &  i > j \neq n, \\
        1, &  j = n,\\
    \end{array}
  \right .
  \quad i = 1,2, ..., n ,
  \quad j = 1,2, ..., n
\end{aligned}$

и правой частью

$\begin{aligned} 
  f_i = 1, \quad i = 1,2, ..., n
\end{aligned}$

при различных $n$.

Решите также эту задачу с помощью библиотеки SciPy.
""")

if chart_visual == 'Матрица':
  st.header('Матрица')
#  if (st.button('Матрица')):
  n = st.slider('N', 0, 1000, 8)
  st.write("N = ", n)
    
  if (st.button('Результат')):
    import scipy as sp
    A = - sp.ones((n, n), 'float')
    for i in range(0, n):
      A[i, i] = 1
      A[i, n - 1] = 1
      if i < n - 1:
        A[i, i + 1: n - 1] = 0
    st.write('A:\n', A)

if chart_visual == 'Описание алгоритма':
  st.header('Описание алгоритма')
  st.write(r"""
$LU$- разложение — это представление матрицы A в виде A = L * U, 
где L — нижнетреугольная матрица с еденичной диагональю, 
а U — верхнетреугольная матрица. $LU$- разложение является модификациеё метода Гаусса.
""")
  if (st.button('LU разложение')):
    #from PIL import Image
    #image = Image.open('LU.jpg')
    #st.image(image)
    st.write(r"""
    $LU$-факторизация матрицы
$$
 A = LU
$$
Нижняя ($L$) и верхняя ($U$) треугольные матрицы
$$
 L = \begin{pmatrix}
  l_{11} & 0      & \cdots &  0       \\
  l_{21} & l_{22} & \cdots &  0       \\
  \cdots & \cdots & \cdots &  \cdots  \\
  l_{n1} & l_{n2} & \cdots &  l_{nn}  \\
\end{pmatrix} ,
\quad U = \begin{pmatrix}
  u_{11} & u_{12} & \cdots &  u_{1n}  \\
  0      & u_{22} & \cdots &  u_{2n}  \\
  \cdots & \cdots & \cdots &  \cdots  \\
  0      & 0      & \cdots &  u_{nn}  \\
\end{pmatrix}  
$$ 
при задании $l_{ii}$ или  $u_{ii}$ для $i = 1,2, \ldots, n$ 

Решение системы уравнений $A x = f$ 
$$
 L y = f ,
 \quad U x = y 
$$

$$
 A = LU,
 \quad a_{ij} = \sum_{s=1}^{n} l_{is} u_{sj} = \sum_{s=1}^{\min(i,j)} l_{is} u_{sj}
$$
Строки $1, 2, \ldots, k-1$ для $u$ и столбцы $1, 2, \ldots, k-1$ для $L$ вычислены

При $k=1$ 
$$
 a_{11} = {\color{red} l_{11} u_{11}},
 \quad a_{1j} = l_{11} {\color{red} u_{1j}}, \quad j = 2,3, \ldots, n ,
 \quad a_{i1} = {\color{red} l_{i1}} u_{11}, \quad i = 2,3, \ldots, n  
$$
Для нового $k = 2, 3, \ldots, n$
$$
 a_{kk} = {\color{red} l_{kk} u_{kk}} + \sum_{s=1}^{k-1} l_{ks} u_{sj}
$$
$$
 a_{kj} = l_{kk} {\color{red} u_{kj}} + \sum_{s=1}^{k-1} l_{ks} u_{sj}, \quad j = k+1, k+2, \ldots, n 
$$
$$
 \quad a_{ik} = {\color{red} l_{ik}} u_{kk} + \sum_{s=1}^{k-1} l_{ks} u_{sj}, \quad i = k+1, k+2, \ldots, n  
$$ 
""")

  if (st.button('Модуль LU')):
    st.write(r"""В модуле lu функция decLU() проводит $LU$- разложения входной 
    матрицы A и записывает результат в матрицу $LU$.""")
    code = '''import numpy as np
def decLU(A):
#Returns the decomposition LU for matrix A
  n = len (A)
  LU = np.copy(A)
  for j in range(0, n - 1):
    for i in range(j + 1, n):
      if LU[i, j] != 0.:
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
  return x'''
    st.code(code, language='python')

if chart_visual == 'Код с NumPy':
  st.subheader("Программа, реализующая решение системы линейных алгебраических уравнений на основе $LU$- разложения")

  code = '''import numpy as np
from lu import decLU, solveLU
n = int(input)
A = - np.ones((n, n), 'float')
for i in range(0, n):
    A[i, i] = 1
    A[i, n - 1] = 1
    if i < n - 1:
        A[i, i + 1: n - 1] = 0
print('A:', A)
LU = decLU(A)
print('LU:', LU)
f = np.ones((n), 'float')
print('b:', f)
x = solveLU(A, f)
print('x:' , x)'''
  st.code(code, language='python')
  
  n = st.slider('N', 0, 1000, 8)
  st.write("N = ", n)

  if (st.button('Результат')):

    import numpy as np
    import time 
    t0 = time.time()
    def decLU(A):
    #Returns the decomposition LU for matrix A
      n = len (A)
      LU = np.copy(A)
      for j in range(0, n - 1):
        for i in range(j + 1, n):
          if LU[i, j] != 0.:
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

    import scipy as sp
    from scipy.linalg import lu
    A = - sp.ones((n, n), 'float')
    for i in range(0, n):
      A[i, i] = 1
      A[i, n - 1] = 1
      if i < n - 1:
        A[i, i + 1: n - 1] = 0
    
    st.write ('A:', A)
    
    LU = decLU(A)
    st.write('LU:', LU)
  
    f = np.ones((n))
    st.write('b:', f)

    x = solveLU(A, f)
    st.write('x:', x)
    t1 = time.time() - t0
    st.write('time:\n' , t1)

if chart_visual == 'Код со SciPy':
  st.subheader("Программа, реализующая решение системы линейных алгебраических уравнений на основе $LU$- разложения с использованием SciPy")
  
  st.write(r"""
scipy.linalg.lu(a, permute_l=False, overwrite_a=False, check_finite=True)

Вычисляет сводную $LU$-декомпозицию матрицы.

Разожение:

$$
A = P L U
$$

где P - матрица перестановок, L - нижняя треугольная с единичными диагональными элементами и U - верхняя треугольная
  
$\textbf{Параметры:}$ 

$\bullet$ a: (M, N) array_like

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ Массив для декомпозиции

$\bullet$ permute_l: bool, optional

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ Выполняет умножение $P*L$

$\bullet$ overwrite_a: bool, optional

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ Следует ли перезаписывать данные в $a$ (может повысить производительность)

$\bullet$ check_finite: bool, optional

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ Следует ли проверять, что входная матрица содержит только конечные числа. Отключение может дать прирост производительности, но может привести к проблемам (сбоям, невозможности завершения), если входные данные содержат бесконечности """)
  
  code = '''import numpy as np
import scipy as sp
from scipy.linalg import lu
A = - sp.ones((n, n), 'float')
for i in range(0, n):
  A[i, i] = 1
  A[i, n - 1] = 1
  if i < n - 1:
    A[i, i + 1: n - 1] = 0
    
print ('A:', A)
    
permute_mat,lower_tri,upper_tri = lu(A)
print('LU:', upper_tri)
  
f = sp.ones((n))
print('b:', f)
  
A1 = sp.linalg.inv(A)
x = sp.dot(A1, f)
print('x:', x)'''
  
  st.code(code, language='python')

if chart_visual == 'Параметрические расчеты со SciPy':
  st.header('Параметрические расчеты со SciPy')
  
  n = st.slider('N', 0, 1000, 8)
  st.write("N = ", n)

  if (st.button('Результат')):
    import numpy as np
    import scipy as sp
    from scipy.linalg import lu
    import time 
    t0 = time.time()
    A = - sp.ones((n, n), 'float')
    for i in range(0, n):
      A[i, i] = 1
      A[i, n - 1] = 1
      if i < n - 1:
        A[i, i + 1: n - 1] = 0
    
    permute_mat,lower_tri,upper_tri = lu(A)
  
    f = sp.ones((n))
  
    A1 = sp.linalg.inv(A)
    x = sp.dot(A1, f)

    t1 = time.time() - t0

    st.write('A:\n', A)
    st.write('LU:\n', upper_tri)
    st.write('b:\n', f)
    st.write('x:\n' , x)
    st.write('time:\n' , t1)

if chart_visual == 'Сравнение':
  st.header('Сравнение')
  n = st.slider('N', 0, 1000, 10)
  st.write("N = ", n)
  if (st.button('Результат')):
    import numpy as np
    import time 
    t0 = time.time()
    def decLU(A):
    #Returns the decomposition LU for matrix A
      n = len (A)
      LU = np.copy(A)
      for j in range(0, n - 1):
        for i in range(j + 1, n):
          if LU[i, j] != 0.:
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

    import scipy as sp
    from scipy.linalg import lu
    A = - sp.ones((n, n), 'float')
    for i in range(0, n):
      A[i, i] = 1
      A[i, n - 1] = 1
      if i < n - 1:
        A[i, i + 1: n - 1] = 0
    
    st.header('NumPy')
    st.write ('A:', A)
    
    LU = decLU(A)
    st.write('LU:', LU)
  
    f = np.ones((n))
    st.write('b:', f)

    x = solveLU(A, f)
    st.write('x:', x)
    t1 = time.time() - t0
    st.write('Time:\n' , t1)

    import numpy as np
    import scipy as sp
    from scipy.linalg import lu
    import time 
    t2 = time.time()
    A = - sp.ones((n, n), 'float')
    for i in range(0, n):
      A[i, i] = 1
      A[i, n - 1] = 1
      if i < n - 1:
        A[i, i + 1: n - 1] = 0
    
    permute_mat,lower_tri,upper_tri = lu(A)
  
    f = sp.ones((n))
  
    A1 = sp.linalg.inv(A)
    x = sp.dot(A1, f)

    t3 = time.time() - t2

    st.header('SciPy')
    st.write('A:\n', A)
    st.write('LU:\n', upper_tri)
    st.write('b:\n', f)
    st.write('x:\n' , x)
    st.write('Time:\n' , t3)

    st.header('Сравнение NumPy и SciPy')
    st.write('Compsrison time (time SciPy - time NumPy):\n' , t3 - t1)
