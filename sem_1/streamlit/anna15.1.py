import streamlit as st
import math as mt

chart_visual = st.sidebar.radio('Содержание', 
    ('Главная', 'Постановка задачи', 'Описание алгоритма', 'Код с NumPy', 'Параметрические расчеты с NumPy', 'Код со SciPy', 'Параметрические расчеты со SciPy', 'Сравнение'))

if chart_visual == 'Главная':
  st.header("Краевые задачи для эллиптических уравнений")
  st.header("Задание 15.1")
  st.subheader("Подготовила студентка первого курса магистратуры, группы СТФИ-122")
  st.subheader("Волкова Анна")

if chart_visual == 'Постановка задачи':
  st.header('Постановка задачи')
  st.write(r"""Напишите программу численного решения на равномерной
по каждому направлению сетке задачи Дирихле для уравнения
Пуассона в прямоугольнике методом разделения переменных.
Для прямого и обратного преобразований Фурье используйте
возможности пакета  $\textsf{NumPy}$.
Продемонстрируйте работоспособность этой программы
при решении краевой задачи

$\begin{aligned}
  -\sum_{\alpha =1}^{2}
  \frac{\partial^2 u}{\partial x^2_\alpha} =
  \frac{32}{l_1^2 l_2^2} \big (x_1 (l_1-x_1) + x_2 (l_2-x_2) \big ),
  \quad {\bm x} \in \Omega ,
\end{aligned}$

$\begin{aligned}
  u({\bm x}) = 0,
  \quad {\bm x} \in \partial \Omega .
\end{aligned}$

Решите также эту задачу с помощью библиотеки SciPy.

""")

if chart_visual == 'Описание алгоритма':
    st.header('Описание алгоритма')
    st.subheader('Теорема сравнения')
    st.write(r"""
Сеточные задачи
$$
  S y ({\bm x}) = \varphi ({\bm x}),
  \quad {\bm x} \in \omega,
  \quad   y({\bm x}) = \mu ({\bm x}),
  \quad {\bm x} \in \partial \omega
$$
$$
  S w ({\bm x}) = \phi({\bm x}),
  \quad {\bm x} \in \omega,
  \quad   w({\bm x}) = \nu({\bm x}),
  \quad {\bm x} \in \partial \omega
$$
Пусть
$$
  |\varphi({\bm x})| \leq \phi({\bm x}),
  \quad {\bm x} \in \omega
$$
$$
  |\mu({\bm x})| \leq \nu({\bm x}),
  \quad {\bm x} \in \partial \omega 
$$
Тогда справедлива оценка
$$
  |y({\bm x})| \leq w({\bm x}),
  \quad {\bm x} \in \omega \cup  \partial \omega 
$$
($w({\bm x})$ $-$ мажорантная функция)

Следствие: для решения однородного уравнения ($\varphi({\bm x}) = 0,$ ${\bm x} \in \omega$) 
$$
  \max_{{\bm x} \in \omega} |y({\bm x})| \leq
  \max_{{\bm x} \in \partial \omega} |\mu({\bm x})| 
$$

""")
    st.subheader('Задачи Дирихле для уравнения Пуассона')
    st.write(r"""
Разностная задача
$$
  - y_{\overline{x}_1x_1} - y_{\overline{x}_2x_2} =
  \varphi({\bm x}),
  \quad {\bm x} \in \omega 
$$
$$
 y({\bm x}) = \mu ({\bm x}),
 \quad {\bm x} \in \partial \omega
$$
Подробная запись
$$
 - y_{\overline{x}_1x_1} - y_{\overline{x}_2x_2} =
 - \frac{1}{h_1^2} (y(x_1+h_1,x_2) - 2y(x_1,x_2) + y(x_1-h_1,x_2) 
$$
$$
 - \frac{1}{h_2^2} (y(x_1,x_2+h_2) - 2y(x_1,x_2) + y(x_1,x_2-h_2)
$$

Для погрешности решения
$z({\bm x}) = y({\bm x}) - u({\bm x}),$ ${\bm x} \in  \omega \cup  \partial \omega $ 
$$
  - z_{\overline{x}_1x_1} - z_{\overline{x}_2x_2} =
  \psi({\bm x}),
  \quad {\bm x} \in \omega 
$$
$$
  z({\bm x}) = 0,
  \quad {\bm x} \in \partial \omega
$$
Погрешность аппроксимации $\psi({\bm x}) = \mathcal{O}(h_1^2 + h_2^2)$ 
""")
    st.subheader('Сходимость приближенного решения')
    st.write(r"""
Расчетная область 
$$
\Omega = \{ {\bm x}~ |~ {\bm x} = (x_1,x_2),~ 0 < x_\alpha <
$$
Мажорантная функции
$$
  w({\bm x}) = \frac{1}{4}(l_1^2+l_2^2 - x_1^2 - x_2^2)
  \| \psi({\bm x})\|_{\infty}
$$
где
$$
  \|v({\bm x})\|_{\infty} =
  \max_{{\bm x} \in \omega} |v({\bm x})| 
$$
Оценка для погрешности 
$$
  \|y({\bm x}) - u({\bm x}) \|_{\infty} \leq
  \frac{1}{4}(l_1^2+l_2^2) \| \psi({\bm x})\|_{\infty} 
$$
Разностная схема сходится в
$L_{\infty}(\omega)$ со вторым порядком по $|h| = (h_1^2 + h_2^2)^{1/2}$ 

""")
    st.subheader('Метод разделения переменных')
    st.write(r"""
Приближенное  решение  
$$
  y({\bm x}) = \sum_{k=1}^{N_1-1} c^{(k)}(x_2) v^{(k)}(x_1),
  \quad {\bm x} \in \omega 
$$
$\varphi^{(k)}(x_2)$ $-$  коэффициенты Фурье правой части
$$
  \varphi^{(k)}(x_2) = \sum_{k=1}^{N_1-1} \varphi({\bm x})
  v^{(k)}(x_1) h_1 
$$

Для $c^{(k)} (x_2)$ 
$$
  - c^{(k)}_{\overline{x}_2x_2} -  \lambda c^{(k)} = \varphi^{(k)}(x_2) ,
  \quad x_2 \in \omega_2
$$
$$
  c^{(k)}_0 = 0,
  \quad c^{(k)}_{N_2} =0.
$$
При каждом $k = 1,2,\ldots,N_1 -1$ трехдиагональная матриуа 
(метод Гаусса, метод прогонки)

Суммирование рядов Фурье $-$ быстрое преобразование Фурье

Трудозатраты $-$ $\mathcal{O} (N_1 N_2 log N_1)$ 

$$
""")

if chart_visual == 'Код с NumPy':
    st.header('Код с NumPy')
    st.write(r"""
Трехдиагональная матрица $A$ задается тремя диагоналями: 

$a_i = a_{ii}$, $\;$ $b_i = a_{i,i+1}$, $\;$ $c_i = a_{i,i-1}$

В модуле $lu3$ функция $decLU3()$ предназначена для $LU$-разложение 
трехдиагоналыюй матрицы $A$. Результат записан в виде трехдиагоналей, причем

$d_i = l_{ii}$, $\;$ $u_i = u_{i,i+1}$, $\;$ $l_i = l_{i,i-1}$

Для решения системы уравнений используется функция $solveLU3()$.

""")
    st.subheader("Модуль lu3")
    code = '''
import numpy as np
def decLU3(a, b, с):
    """
    Input of tridiagonal matrix A:
        a[i] = A[i, i],
        b[i] = A[i, i + 1],
        c[i] = A[i, i - 1]
    Returns the decompositon LU for tridiagonal matrix:
        d[i] = L[i, i]
        u[i] = U[i, i + 1]
        l[i] = L[i, i - 1]
    """
    n = len(a)
    d = np.copy(a)
    u = np.copy(b)
    l = np.copy(с)
    for i in range(1, n):
        al = l[i] / d[i - 1]
        d[i] = d[i] - al * u[i - 1]
        l[i] = al
    return d, u, l
def solveLU3(a, b, c, f ):
    """
    Solve the linear system Ax = b with tridiagonal matrix:
        a[i] = A[i, i],
        b[i] = A[i, i + 1],
        с[i] = A[i, i - 1]
    """
    n = len(a)
    # LU decomposition
    d, u, l = decLU3(a, b, c)
    x = np.copy(f)
    # forward substitution process
    for i in range(1, n):
        x[i] = x[i] - l[i] * x[i - 1]
    # back substitution process
    x[n - 1] = x[n - 1] / d[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (x[i] - u[i] * x[i + 1]) / d[i] 
    return x
'''
    st.code(code, language='python')

    st.write(r"""
Синус-преобразование Фурье проводится с помощью функций $fft.rfft()$ (прямое преобразование) и
$fft.irfft()$ (обратное преобразование) пакета $NumPy$. 
Исходная веществен­ ная сеточная функция продлевается нечетным образом, 
к которой затем при­ меняется дискретное преобразование Фурье. 
В модуле $fftPoisson$ функция $fftPoisson()$ реализует численное решение сеточной э
ллиптической задачи разделением переменных с преобразованием Фурье по первой переменной.
Ре­шение систем уравнений с трехдиагоналыюй матрицей обеспечивается функ­цией $solveLU3()$ модуля $luЗ$.
    """)

    st.subheader("Модуль fftРоissоп")
    st.write(r"Приближенное вычисление интеграла на интервале $[a,b]$")
    code = '''
import numpy as np
import math as mt
from lu3 import solveLU3

def fftPoisson(f, l1, l2, n1, n2):
    """
Numerical Solution of the Dirichlet problem 
for Poisson equation in a rectangle.
Fast Fourier transform in the variable x.
    """
    h1 = l1 / n1
    h2 = l2 / n2
    # Fourier coefficients of the right side
    y = np.zeros((n1 + 1, n2 + 1), 'float')
    r = np.zeros((2 * n1), 'float')
    tt = np.zeros((n1 + 1), 'cfloat')
    for j in range(1, n2):
        for i in range(1, n1): 
            r[i] = f(i * h1, j * h2)
            r[2 * n1 - i] = - r[i]
        rt = np.fft.rfft(r).imag
        y [0 : n1 + 1, j] = rt [0 : n1 + 1]
    # Fourier coefficients for the solution,
    a = np.ones((n2 + 1), 'float') 
    b = np.zeros((n2 + 1), "float")
    c = np.zeros((n2 + 1), "float")
    q = np.zeros((n2 + 1), "float")
    for i in range(1, n1):
        for j in range(1, n2):
            a[j] = 2. + (2. * mt.sin(i * mt .pi / (2 * n1))/h1) ** 2 * h2 ** 2
            b[j] = - 1.
            c[j] = - 1.
            q[j] = y[i, j] * h2 ** 2
        p = solveLU3(a, b, c, q)
        y[i,:] = p
    # Inverse Fourier transform, 
    for j in range(1, n2):
        for i in range(1, n1): 
            tt[i] = y[i, j] * 1.j
        yt = np.fft.irfft(tt)
        y[0 : n1, j] = yt[0 : n1]
    return y
    '''
    st.code(code, language='python')

    st.subheader("Основной код")
    st.write(r"""Решение модельной задами с точным решением
$\begin{aligned}
  \  {\bm u(x)} =
  \frac{16}{l_1^2 l_2^2} \big (x_1 (l_1-x_1) x_2 (l_2-x_2) \big )
\end{aligned}$

обеспечивается следующей программой.
    """)
    code = '''
import numpy as np
from fftPoisson import fftPoisson 
import matplotlib.pyplot as plt 
l1 = 2.
l2 = 1.
def f(x,y):
    return 32. * (x * (l1 - x) + y * (l2 - y)) / (l1 * l2) ** 2
n1 = 32
n2 = 16
y = fftPoisson(f, l1, l2, n1, n2).T
print ('max у =', np.amax(y))
x1 = np.linspace(0., l1, n1 + 1)
x2 = np.linspace(0., l2, n2 + 1)
X1, X2 = np.meshgrid(x1, x2)
plt.contourf(X1, X2, y, cmap = plt.cm.gray)
plt.colorbar()
plt.show()
  '''
    st.code(code, language = 'python')

if chart_visual == 'Параметрические расчеты с NumPy':
    st.header('Параметрические расчеты с NumPy')

    st.write(r"""$l2$, $l1$:""")
    l2, l1 = st.slider(' ', 0.0, 5.0, (1., 2.), 0.01)

    #Модуль Lu3
    import numpy as np
    def decLU3(a, b, с):
        """
        Input of tridiagonal matrix A:
            a[i] = A[i, i],
            b[i] = A[i, i + 1],
            c[i] = A[i, i - 1]
        Returns the decompositon LU for tridiagonal matrix:
            d[i] = L[i, i]
            u[i] = U[i, i + 1]
            l[i] = L[i, i - 1]
        """
        n = len(a)
        d = np.copy(a)
        u = np.copy(b)
        l = np.copy(с)
        for i in range(1, n):
            al = l[i] / d[i - 1]
            d[i] = d[i] - al * u[i - 1]
            l[i] = al
        return d, u, l
    def solveLU3(a, b, c, f ):
        """
        Solve the linear system Ax = b with tridiagonal matrix:
            a[i] = A[i, i],
            b[i] = A[i, i + 1],
            с[i] = A[i, i - 1]
        """
        n = len(a)
        # LU decomposition
        d, u, l = decLU3(a, b, c)
        x = np.copy(f)
        # forward substitution process
        for i in range(1, n):
            x[i] = x[i] - l[i] * x[i - 1]
        # back substitution process
        x[n - 1] = x[n - 1] / d[n - 1]
        for i in range(n - 2, -1, -1):
            x[i] = (x[i] - u[i] * x[i + 1]) / d[i] 
        return x

    import numpy as np
    import math as mt
#    from lu3 import solveLU3

    def fftPoisson(f, l1, l2, n1, n2):
        """
Numerical Solution of the Dirichlet problem 
for Poisson equation in a rectangle.
Fast Fourier transform in the variable x.
        """
        h1 = l1 / n1
        h2 = l2 / n2
        # Fourier coefficients of the right side
        y = np.zeros((n1 + 1, n2 + 1), 'float')
        r = np.zeros((2 * n1), 'float')
        tt = np.zeros((n1 + 1), 'cfloat')
        for j in range(1, n2):
            for i in range(1, n1): 
                r[i] = f(i * h1, j * h2)
                r[2 * n1 - i] = - r[i]
            rt = np.fft.rfft(r).imag
            y [0 : n1 + 1, j] = rt [0 : n1 + 1]
        # Fourier coefficients for the solution,
        a = np.ones((n2 + 1), 'float') 
        b = np.zeros((n2 + 1), "float")
        c = np.zeros((n2 + 1), "float")
        q = np.zeros((n2 + 1), "float")
        for i in range(1, n1):
            for j in range(1, n2):
                a[j] = 2. + (2. * mt.sin(i * mt .pi / (2 * n1))/h1) ** 2 * h2 ** 2
                b[j] = - 1.
                c[j] = - 1.
                q[j] = y[i, j] * h2 ** 2
            p = solveLU3(a, b, c, q)
            y[i,:] = p
        # Inverse Fourier transform, 
        for j in range(1, n2):
            for i in range(1, n1): 
                tt[i] = y[i, j] * 1.j
            yt = np.fft.irfft(tt)
            y[0 : n1, j] = yt[0 : n1]
        return y

    import numpy as np
    #from fftPoisson import fftPoisson 
    import matplotlib.pyplot as plt 
#    l1 = 2.
#    l2 = 1.
    def f(x,y):
        return 32. * (x * (l1 - x) + y * (l2 - y)) / (l1 * l2) ** 2
    n1 = 32
    n2 = 16
    y = fftPoisson(f, l1, l2, n1, n2).T
    print ('max у =', np.amax(y))
    x1 = np.linspace(0., l1, n1 + 1)
    x2 = np.linspace(0., l2, n2 + 1)
    X1, X2 = np.meshgrid(x1, x2)
    plt.contourf(X1, X2, y, cmap = plt.cm.gray)
    plt.colorbar()
    plt.show()

    fig, ax = plt.subplots()
    ax.contourf(X1, X2, y)
#    ax.set_xlim(xmin=0, xmax=2)
    ax.grid()
    st.pyplot(fig)

    st.write('max у =', np.amax(y))

if chart_visual == 'Код со SciPy':
    st.header('Код со SciPy')

    st.subheader('')
    st.write(r"""
    
    """)

    st.write(r" ")
    code = '''

    '''
    st.code(code, language='python')

if chart_visual == 'Параметрические расчеты со SciPy':
    st.header('Параметрические расчеты со SciPy')
    
    st.write(r"""$l2$, $l1$:""")
    l2, l1 = st.slider(' ', 0.0, 5.0, (1., 2.), 0.01)


if chart_visual == 'Сравнение':
    st.header('Сравнение')
    st.write(r"""$l2$, $l1$:""")
    l2, l1 = st.slider(' ', 0.0, 5.0, (1., 2.), 0.01)

        #Модуль Lu3
    import numpy as np
    import time 
    time_0 = time.time()
    def decLU3(a, b, с):
        """
        Input of tridiagonal matrix A:
            a[i] = A[i, i],
            b[i] = A[i, i + 1],
            c[i] = A[i, i - 1]
        Returns the decompositon LU for tridiagonal matrix:
            d[i] = L[i, i]
            u[i] = U[i, i + 1]
            l[i] = L[i, i - 1]
        """
        n = len(a)
        d = np.copy(a)
        u = np.copy(b)
        l = np.copy(с)
        for i in range(1, n):
            al = l[i] / d[i - 1]
            d[i] = d[i] - al * u[i - 1]
            l[i] = al
        return d, u, l
    def solveLU3(a, b, c, f ):
        """
        Solve the linear system Ax = b with tridiagonal matrix:
            a[i] = A[i, i],
            b[i] = A[i, i + 1],
            с[i] = A[i, i - 1]
        """
        n = len(a)
        # LU decomposition
        d, u, l = decLU3(a, b, c)
        x = np.copy(f)
        # forward substitution process
        for i in range(1, n):
            x[i] = x[i] - l[i] * x[i - 1]
        # back substitution process
        x[n - 1] = x[n - 1] / d[n - 1]
        for i in range(n - 2, -1, -1):
            x[i] = (x[i] - u[i] * x[i + 1]) / d[i] 
        return x

    import numpy as np
    import math as mt
#    from lu3 import solveLU3

    def fftPoisson(f, l1, l2, n1, n2):
        """
Numerical Solution of the Dirichlet problem 
for Poisson equation in a rectangle.
Fast Fourier transform in the variable x.
        """
        h1 = l1 / n1
        h2 = l2 / n2
        # Fourier coefficients of the right side
        y = np.zeros((n1 + 1, n2 + 1), 'float')
        r = np.zeros((2 * n1), 'float')
        tt = np.zeros((n1 + 1), 'cfloat')
        for j in range(1, n2):
            for i in range(1, n1): 
                r[i] = f(i * h1, j * h2)
                r[2 * n1 - i] = - r[i]
            rt = np.fft.rfft(r).imag
            y [0 : n1 + 1, j] = rt [0 : n1 + 1]
        # Fourier coefficients for the solution,
        a = np.ones((n2 + 1), 'float') 
        b = np.zeros((n2 + 1), "float")
        c = np.zeros((n2 + 1), "float")
        q = np.zeros((n2 + 1), "float")
        for i in range(1, n1):
            for j in range(1, n2):
                a[j] = 2. + (2. * mt.sin(i * mt .pi / (2 * n1))/h1) ** 2 * h2 ** 2
                b[j] = - 1.
                c[j] = - 1.
                q[j] = y[i, j] * h2 ** 2
            p = solveLU3(a, b, c, q)
            y[i,:] = p
        # Inverse Fourier transform, 
        for j in range(1, n2):
            for i in range(1, n1): 
                tt[i] = y[i, j] * 1.j
            yt = np.fft.irfft(tt)
            y[0 : n1, j] = yt[0 : n1]
        return y

    import numpy as np
    #from fftPoisson import fftPoisson 
    import matplotlib.pyplot as plt 
#    l1 = 2.
#    l2 = 1.
    def f(x,y):
        return 32. * (x * (l1 - x) + y * (l2 - y)) / (l1 * l2) ** 2
    n1 = 32
    n2 = 16
    y = fftPoisson(f, l1, l2, n1, n2).T
    print ('max у =', np.amax(y))
    x1 = np.linspace(0., l1, n1 + 1)
    x2 = np.linspace(0., l2, n2 + 1)
    X1, X2 = np.meshgrid(x1, x2)
    plt.contourf(X1, X2, y, cmap = plt.cm.gray)
    plt.colorbar()
    plt.show()

    fig, ax = plt.subplots()
    ax.contourf(X1, X2, y)
#    ax.set_xlim(xmin=0, xmax=2)
    ax.grid()
    st.pyplot(fig)

    st.write('max у =', np.amax(y))
    time_numpy = time.time() - time_0
    st.write ('Время NumPy=', time_numpy)



    import numpy as np
    import pandas as pd

    t_ol = np.transpose(t_numpy)
    y_new = np.transpose(y_numpy)
    y_del = y_new - sol.y
  
    mat = np.vstack((t_ol, y_new, sol.y, y_del))
    mat1 = np.transpose(mat)
    df = pd.DataFrame(
        mat1,
        columns=('t', 'y (NumPy)', 'dy/dt (NumPy)', 'y (SciPy)', 'dy/dt (SciPy)', 'Сравнение y', 'Сравнение dy/dt'))  
  
    st.table(df)
