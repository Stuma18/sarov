
import streamlit as st

chart_visual = st.sidebar.radio('Содержание', 
    ('Главная', 'Постановка задачи', 'Описание алгоритма', 'Код с NumPy', 'Код со SciPy', 'Параметрические расчеты со SciPy', 'Сравнение', 'Выводы', 'Спасибо'))

if chart_visual == 'Главная':
  st.header("Нелинейные уравнения и системы")
  st.header("Задание 8.2")
  st.subheader("Подготовили студентки первого курса магистратуры, группы СТФИ-122")
  st.subheader("Студеникина Мария")
  st.subheader("Волкова Анна")
  st.subheader("Савкина Виктория")

if chart_visual == 'Постановка задачи':
  st.header('Постановка задачи')
  st.write(r"""
Напишите программу для нахождения решения системы нелинейных уравнений
$F(x) = 0$ методом Ньютона при численном вычислении матрицы Якоби.

С ее помощью найдите приближенное решение системы

$\begin{aligned}
  (3 + 2x_1) x_1 - 2 x_2 = 3,
\end{aligned}$

$\begin{aligned}
  (3 + 2x_i) x_i - x_{i-1} - 2 x_{i+1} = 2,
  \quad i = 2, 3, ..., n-1,
\end{aligned}$

$\begin{aligned}
  (3 + 2x_n) x_n - x_{n-1} = 4
\end{aligned}$

и сравните его с точным решением
$x_i = 1, \ i = 1,2,..., n$
при различных $n$.

Решите также эту задачу с помощью библиотеки SciPy.
""")

if chart_visual == 'Описание алгоритма':
    st.header('Описание алгоритма')
    st.write(r"""
Для решения этой задачи сперва необходимо вычислить матрицу Якоби, которая состоит из частных производных. 
В данной задаче матрица будет высчитываться с помощью конечных разностей
""")
    #from PIL import Image
    #image = Image.open('new1.png')
    #st.image(image)
    st.write(r"""F'(x) = $\begin{pmatrix}
    {\displaystyle
    \frac{\partial f_1(x)}{\partial x_1}} &
    {\displaystyle \frac{\partial f_1(x)}{\partial x_2} }&
    \cdots &
    {\displaystyle \frac{\partial f_1(x)}{\partial x_n} }\\[5pt]
    {\displaystyle
    \frac{\partial f_2(x)}{\partial x_1} }&
    {\displaystyle \frac{\partial f_2(x)}{\partial x_2} }&
    \cdots &
    {\displaystyle \frac{\partial f_2(x)}{\partial x_n} }\\[0pt]
    {\displaystyle
    \cdots }& \cdots & \cdots & \cdots \\[0pt]
    {\displaystyle
    \frac{\partial f_n(x)}{\partial x_1} }&
    {\displaystyle \frac{\partial f_n(x)}{\partial x_2} }&
    \cdots &
    {\displaystyle \frac{\partial f_n(x)}{\partial x_n} }\
    \end{pmatrix}$
    """)
        
    st.write(r"""
Частные производные высчитываются по формуле
$\begin{aligned}
f' (x) \approx \frac{f(x + h) - f(x)}{h}
\end{aligned}$
, где $h$ — это  небольшое число. Если оно слишком большое, будут большие ошибки усечения. Если оно слишком маленькое, то будут большие ошибки округления. 
Обычно его необходимо калибровать методом проб и ошибок, в данной задаче берется 

$\begin{aligned}
  h = 1.0e^{-4}
\end{aligned}$
""")
    if (st.button('Матрица Якоби')):
        code = '''import numpy as np
import math as mt
from lu import solveLU

def jacobian(f, x):
    #Вычисление якобиана с использованием конечных разностей
    h = 1.0e-4
    n = len(x)
    Jac = np.zeros((n,n), 'float') 
    fO = f(x)
    for i in range(n):
        tt = x[i]
        x[i] = tt + h
        f1 = f(x)
        x[i] = tt
        Jac[:,i] = (f1 - f0) / h
    return Jac, fO'''
    
        st.code(code, language='python')

    if (st.button('Метод Ньютона')):
        st.write(r"""
Для решения системы нелинейных уравнений F(x) = 0 методом Ньютона используется LU - разложние из модуля LU
    """)
        code = '''
    def newton(f, x, tol=1.0e-9):
    #Решает уравнения системы f(x) =0 с помощью
    #метод Ньютона, использующий {x} в качестве начального предположения
    #Решение линейной системы Ax = b по модулю lu
    iterMax = 50
    for i in range(iterMax):
        Jac, fO = jacobian(f, x)
        if mt.sqrt(np.dot(fO, fO) / len(x)) < tol:
            return x, i
        dx = solveLU(Jac, fO) 
        x = x - dx

print ("Too many iterations for the Newton method")'''
        st.code(code, language='python')
    
    if (st.button('Модуль LU')):
        code = '''import numpy as np

def decLU(A):
#Возвращает значение разложения LU для матрицы A
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
    #Решение линейную систему Ax = b
      n = len (A)
    #Разложение LU
      LU = decLU(A)
      x = np.copy(f)
    #прямая замена
      for i in range(1, n):
        x[i] = x[i] - np.dot(LU[i, 0:i], x[0:i])
    #процесс обратной замены
      for i in range(n-1, -1, -1):
        x[i] = (x[i] - np.dot(LU[i, i+1:n], x[i+1:n])) / LU[i, i]
      return x'''
        st.code(code, language='python')
    
if chart_visual == 'Код с NumPy':
    st.header('Код с NumPy')
    code = '''import numpy as np
import newton

n = int(input())

def f(x):
    f = np.zeros((n), 'float') 
    for i in range (1, n - 1):
        f [i] = (3 + 2 * x [i]) * x [i] - x [i - 1] - 2 * x [i + 1] - 2 
    f [O] = (3 + 2 * x [0]) * x [0] - 2 * x [1] - 3
    f [n - 1] = (3 + 2 * x [n - 1] ) * x [n - 1] - x [n - 2] - 4
    return f

xO = np.zeros((n), 'float')
x, iter = newton(f, xO)

print ('Newton iteration = ', iter)
print ('Solution:', x)'''
    st.code(code, language='python')

    n = st.slider('N', 1, 400, 10)
    st.write("N = ", n)

    if (st.button('Результат')):
        from numpy import*
        import time 
        t0 = time.time()
        def jacobian(f, x):
            h = 1.0e-4
            n = len(x)
            Jac = zeros([n,n])
            f0 = f(x)
            for i in arange(0,n,1):
                tt = x[i]
                x[i] = tt + h
                f1= f(x)
                x[i] = tt
                Jac [:,i] = (f1 - f0)/h
            return Jac, f0
        
        def newton(f, x, tol=1.0e-9):
            iterMax = 50
            for i in range(iterMax):
                Jac, fO = jacobian(f, x)
                if sqrt(dot(fO, fO) / len(x)) < tol:
                    return x, i                 
                dx = linalg.solve(Jac, fO)
                x = x - dx
            print ("Too many iterations for the Newton method")
        
        def f(x):
            f = zeros([n])
            for i in arange(0,n-1,1):
                f[i] = (3 + 2 * x[i]) * x[i] - x[i-1] - 2 * x[i+1] - 2
            f [0] = (3 + 2 * x[0]) * x[0] - 2*x[1] - 3
            f[n-1] = (3 + 2 * x[n-1]) * x[n-1] - x[n-2] - 4
            return f
        x0 =zeros([n])
        x, iter = newton(f, x0)
        t1 = time.time() - t0
        st.write ('Решение:\n', x)
        st.write ('Колличество итераций: ', iter)
        st.write('Время выполнения:\n' , round(t1, 5))

if chart_visual == 'Код со SciPy':
    st.header('Код со SciPy')
    if (st.button('Метод Крылова')):
        st.write(r"""
Библиотечная функция scipy.optimize.root выбрана, потому что имеет обширную библиотеку методов.

Методы broyden1, broyden2, anderson, linearmixing, diagbroyden, excitingmixing, krylov являются точными методами Ньютона. 
""")
    
#        from PIL import Image
#        image = Image.open('new4.jpg')
#        st.image(image)
    if (st.button('Код')):
        code = '''from numpy import
from scipy import optimize

def f(x):
    f = zeros([n])
    for i in arange(0, n - 1, 1):
        f [i] = (3 + 2 * x [i]) * x [i] - x [i - 1] - 2 * x [i + 1] - 2
    f [0] = (3 + 2 * x [0] ) * x [0] - 2 * x [1] - 3
    f [n - 1] = (3 + 2 * x [n - 1] ) * x [n - 1] - x [n - 2] - 4
        return f
    x0 = zeros([n])

    sol = optimize.root(f, x0, method = 'krylov')
    
print('Solution:', sol.x)
print('Krylov method iteration = ', sol.nit)
    '''
        st.code(code, language='python')
    
    if (st.button('scipy.optimize.root')):
        #from PIL import Image
        #image = Image.open('new5.png')
        #st.image(image)
        #image = Image.open('new6.png')
        #st.image(image)

        #st.write("scipy.optimize.root(fun, x0, args=(), method='hybr', jac=None, tol=None, callback=None, options=None) Find a root of a vector function.")
        st.write("scipy.optimize.root(fun, x0, args=(), method='krylov', tol=None, callback=None, options={})")
        st.write(r"""
Находит корень векторной функции

$\textbf{Параметры:}$ 

$\bullet$ func:

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ Векторная функция для нахождения корня 

$\bullet$ x0:

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ Первоначальное предположение

$\bullet$ args:

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ Дополнительные аргументы, передаваемые целевой функции и ее якобиану

$\bullet$ method:

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ Тип решателя. Должен быть одним из

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ $\bullet$ 'hybr'

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ $\bullet$ 'lm'

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ $\bullet$ 'broyden1'

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ $\bullet$ 'broyden2'

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ $\bullet$ 'anderson'

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ $\bullet$ 'linearmixing'

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ $\bullet$ 'diagbroyden'

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ $\bullet$ 'excitingmixing'

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ $\bullet$ 'krylov'

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ $\bullet$ 'df-sane'

$\bullet$ jac:

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ Если jac является логическим значением и имеет значение True, предполагается, что fun возвращает
значение Якобиана вместе с целевой функцией. Если значение False, то Якобиан будет оценен численно

$\bullet$ tol:

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ Допуск к прекращению. Для детального управления используйте параметры, специфичные для решателя

$\bullet$ callback:

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ Дополнительная функция обратного вызова. Он вызывается на каждой итерации как
обратный вызов (x, f), где x - текущее решение, а f - соответствующий остаток. Для всех методов, кроме 'hybr' и 'lm'.

$\bullet$ options:

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ Словарь параметров решателя

$\textbf{Результаты:}$ 

$\bullet$  sol: OptimizeResult

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ Решение представлено в виде объекта OptimizeResult. Важными атрибутами являются: x - массив решений, success - Логический флаг, указывающий, успешно ли завершен алгоритм, и сообщение, описывающее причину завершения

""")

if chart_visual == 'Параметрические расчеты со SciPy':
  st.header('Параметрические расчеты со SciPy')
  
  n = st.slider('N', 1, 400, 10)
  st.write("N = ", n)

  if (st.button('Результат')):
    from numpy import*
    from scipy import optimize
    import time 
    t0 = time.time()
    def f(x):
        f = zeros([n])
        for i in arange(0,n-1,1):
            f [i] = (3 + 2 * x [i]) * x[i] - x[i-1] - 2 * x[i+1] - 2
        f [0] = (3 + 2 * x [0] ) * x [0] - 2 * x[1] - 3
        f [n - 1] = (3 + 2 * x [n - 1] ) * x [n - 1] - x [n - 2] - 4
        return f
    x0 = zeros([n])

    sol = optimize.root(f,x0, method='krylov')
    t1 = time.time() - t0
    st.write('Решение:\n', sol.x)
    st.write('Колличество итераций: ',sol.nit)
    st.write('Время выполнения:\n' , round(t1, 5))

if chart_visual == 'Сравнение':
    st.header('Сравнение')
    n = st.slider('N', 1, 400, 10)
    st.write("N = ", n)
    if (st.button('Результат')):

        from numpy import*
        import time 
        t0 = time.time()
        def jacobian(f, x):
            h = 1.0e-4
            n = len(x)
            Jac = zeros([n,n])
            f0 = f(x)
            for i in arange(0,n,1):
                tt = x[i]
                x[i] = tt + h
                f1= f(x)
                x[i] = tt
                Jac [:,i] = (f1 - f0)/h
            return Jac, f0
        
        def newton(f, x, tol=1.0e-9):
            iterMax = 50
            for i in range(iterMax):
                Jac, fO = jacobian(f, x)
                if sqrt(dot(fO, fO) / len(x)) < tol:
                    return x, i                 
                dx = linalg.solve(Jac, fO)
                x = x - dx
            print ("Too many iterations for the Newton method")
        
        def f(x):
            f = zeros([n])
            for i in arange(0,n-1,1):
                f[i] = (3 + 2*x[i])*x[i] - x[i-1] - 2*x[i+1] - 2
            f [0] = (3 + 2*x[0] )*x[0] - 2*x[1] - 3
            f[n-1] = (3 + 2*x[n-1] )*x[n-1] - x[n-2] - 4
            return f
        x0 =zeros([n])
        x_new, iter = newton(f, x0)
        t_new = time.time() - t0
        #st.header('Метод Ньютона')
        #st.write ('Solution Newton method:\n', x_new)
        #st.write ('Newton iteration = ', iter)
        #st.write('Time Newton method:\n' , t_new)

        from numpy import*
        from scipy import optimize
        import time 
        t0 = time.time()
        def f(x):
            f = zeros([n])
            for i in arange(0,n-1,1):
                f[i] = (3 + 2 * x[i]) * x[i] - x[i-1] - 2 * x[i+1] - 2
            f [0] = (3 + 2 * x[0] ) * x[0] - 2 * x[1] - 3
            f[n-1] = (3 + 2 * x[n-1] ) * x[n-1] - x[n-2] - 4
            return f
        x0 = zeros([n])

        sol = optimize.root(f,x0, method='krylov')
        t_krylov = time.time() - t0
        #st.header('Метод Крылова')
        #st.write('Solution Krylov method:\n', sol.x)
        #st.write('Krylov method iteration = ',sol.nit)
        #st.write('Time Krylov method:\n' , t_krylov)

#        if (st.button('Сравнение')):
        #st.header('Сравнение')
        #st.write('Compsrison (Solution Krylov - Solution Newton):\n', sol.x - x_new)
        #st.write('Compsrison time (time Krylov - time Newton):\n' , t_krylov - t_new)


        x_krylov = sol.x
        com_sol = x_new - sol.x
        com_time = t_new - t_krylov
        com_inter = iter - sol.nit

        import numpy as np
        import pandas as pd
        import math

        mat = np.vstack((x_new, x_krylov, com_sol))
        mat1 = np.transpose(mat)
        df = pd.DataFrame(
            mat1,
            columns=('Метод Ньютона', 'Метод Крылова', 'Сравнение'))    

        #df.loc[ len(df.index) ] = (t_new, t_krylov, com_time)
        #df.loc[ len(df.index) ] = (iter, sol.nit, com_inter)

        mat_2 = np.vstack((t_new, t_krylov, com_time))
        mat_3 = np.vstack((iter, sol.nit, com_inter))
        #mat_3 = np.vstack(((round(iter, 0)), sol.nit), math.trunc(com_inter)))
        mat_4 = np.hstack((mat_2, mat_3))
        mat_5 = np.transpose(mat_4)
        df_1 = pd.DataFrame(
            mat_5,
            index=['Время', 'Колличество итераций'],
            columns=('Метод Ньютона', 'Метод Крылова', 'Сравнение'))    

        st.table(df_1)
        
        st.table(df)

if chart_visual == 'Выводы':
    st.header('Выводы')
    st.write(r"""1) Программа, реализованная методом Крылова, до $n = 378$ выдает приближенное решение, которое сходится с $x_i = 1$.
    Метод Ньютона также успешно реализован в этой программе.
    
    """)
    st.write(r"""
    2) При $ n > 378 $ метод Крылова выдает результат решения с большой ошибкой. Это может быть связанно с автоматической адаптацией к шагу, что становится причиной падения быстродействия.
    """)

if chart_visual == 'Спасибо':
    from PIL import Image
    image = Image.open('new7.jpg')
    st.image(image)
