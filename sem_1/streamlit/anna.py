import streamlit as st
import math as mt

chart_visual = st.sidebar.radio('Содержание', 
    ('Главная', 'Постановка задачи', 'Описание алгоритма', 'Код с NumPy', 'Параметрические расчеты с NumPy', 'Код со SciPy', 'Параметрические расчеты со SciPy', 'Сравнение'))

if chart_visual == 'Главная':
  st.header("Численное интегрирование")
  st.header("Задание 11.2")
  st.subheader("Подготовила студентка первого курса магистратуры, группы СТФИ-122")
  st.subheader("Волкова Анна")

if chart_visual == 'Постановка задачи':
  st.header('Постановка задачи')
  st.write(r"""Напишите программу для приближенного вычисления
интеграла от функции $f(x)$ на интервале $[a,b]$
с использованием квадратурной формулы Гаусса с весом
$\varrho(x) = 1$ на основе табличного задания узлов и коэффициентов.

Используйте эту программу для вычисления интеграла

$\begin{aligned}
  I = \int_{0}^{1} \frac{\ln(x)}{1 - x} dx
\end{aligned}$

при различном числе узлов и сравните приближенное значение
интеграла с точным.

Решите также эту задачу с помощью библиотеки SciPy.
""")

if chart_visual == 'Описание алгоритма':
    st.header('Описание алгоритма')
    st.subheader('Квадратурные формулы')
    st.write(r"""
Приближенное вычисление определенного интеграла
$$
  I = \int_a^b \varrho (x) f(x) dx
$$
на некотором классе функций $f(x)$ с заданной весовой функцией
$\varrho(x)$

Подынтегральная функция задается в отдельных точках отрезка $[a,b]$

$x_i$, $i=0,1,\ldots,n$

Квадратурная формула
$$
  \int_a^b \varrho (x) f(x) dx \approx
  \sum_{i=0}^{n} c_i f(x_i)
$$

$c_i, \ i=0,\ldots,n$ — коэффициенты квадратурной
формулы
""")
    st.subheader('Погрешность квадратурной формулы')
    st.write(r"""
Точность приближенного вычисления интеграла
$$
  \psi = \int_a^b \varrho (x) f(x) dx -
  \sum_{i=0}^{n} c_i f(x_i)
$$

Минимизация погрешности (увеличение точности) 
- за счет выбора коэффициентов квадратурной формулы
- за счет выбора узлов интегрирования
""")
    st.header('Квадратурные формулы Гаусса')
    st.subheader('Квадратурные формулы повышенной точности')
    st.write(r"""
Повышение точности квадратурной формулы
- выбор коэффициентов квадратурной формулы $c_i, \ i =0,1,\ldots,n$
- выбор узлов интерполяции $x_i, \ i =0,1,\ldots,n$

Квадратурные формулы интерполяционного типа 
$$
  \int_a^b \varrho (x) f(x) dx \approx
  \sum_{i=0}^{n} c_i f(x_i)
$$
$$
  c_i = \int_a^b \varrho (x)\frac {\omega(x)}
  {(x-x_i) \omega' (x_i)} dx ,
  i = 0,1,\ldots,n
$$
являются точными для алгебраических полиномов степени $n$

Квадратурные формулы Гаусса за счет выбора узлов интерполирования 
точны для любого алгебраического многочлена степени $2n+1$

""")
    st.subheader('Построение квадратурных формул')
    st.write(r"""
Потребуем, чтобы квадратурная формула была точна для
любого алгебраического многочлена степени $m$

Для функций $f(x) = x^{\alpha}, \ \alpha = 0,1,\ldots,m$:
$$
  \int_a^b \varrho (x) x^{\alpha} dx =
  \sum_{i=0}^{n} c_i x_i^{\alpha},
  \quad i=0,1,\ldots,m 
$$
Для определения $2n+2$ неизвестных $c_i, \ x_i, \ i = 0,1,\ldots,n$ имеем нелинейную систему из $m+1$
уравнений: $m=2n+1$

Узлы из условия, чтобы многочлен
$$
  \omega(x) = \prod_{i=0}^{n} (x-x_i)
$$
был ортогонален с весом $\varrho (x)$ любому многочлену $q(x)$
степени меньше $n+1$
$$
  \int_a^b \varrho (x) \omega(x) q(x) dx = 0
$$
""")
    st.subheader('Примеры формул Гаусса')
    st.write(r"""
Область интегрирования: $a = - 1$, $b = 1$ 

- $n = 0$ 
$$
 c_0 = 2,
 \quad x_0 = 0 
$$ 
- $n = 1$ 
$$
 c_0 = c_1 = 1,
 \quad x_0 = - \frac{1}{\sqrt{3} } 
 \quad x_1 = \frac{1}{\sqrt{3} } 
$$ 
- $n = 2$ 
$$
 c_0 = c_2 = \frac{5}{9} ,
 \quad c_1 = \frac{8}{9} ,
 \quad x_0 = - \frac{\sqrt{3} }{\sqrt{5} } ,
 \quad x_1 = 0 ,
 \quad x_2 = \frac{\sqrt{3} }{\sqrt{5} }
$$
""")

if chart_visual == 'Код с NumPy':
    st.header('Код с NumPy')
    st.write(r"""
В модуле gauss функция gauss() вычисляет приближенное значение интегра­ла по табличным данным.
""")
    st.subheader("Модуль gauss")
    code = '''
import numpy as np
def gauss(f, a, b, n):
        """
Integral of f(x) from a to b computed by 
Gauss-Legendre quadrature using m nodes.
        """
    if n > 8 or n < 2:
        print ('The number of nodes must be greater than 2 and less than 8')
        return 0
    x = np.zeros(n, 'float')
    c = np.zeros(n, 'float')
    if n == 2:
        x[0] = 0.57735027
        x[1] = - x[0]
        c[0] = 1.
        c[1] = c[0]
    if n == 3:
        x[0] = 0.77459667
        x[1] = - x[0]
        x[2] = 0.
        c[0] = 0.55555556
        c[1] = c[0]
        c[2] = 0.88888889
    if n == 4:
        x[0] = 0.86113631
        x[1] = - x[0]
        x[2] = 0.33998104
        x[3] = - x[2]
        c[0] = 0.34785485 
        c[1] = c[0]
        c[2] = 0.65214515
        c[3] = c[2]
    if n == 5:
        x[0] = 0.90617985
        x[1] = - x[0]
        x[2] = 0.53846931
        x[3] = - x[2]
        x[4] = 0.
        c[0] = 0.23692689
        c[1] = c[0]
        c[2] = 0.47862867
        c[3] = c[2]
        c[4] = 0.56888889
    if n == 6:
        x[0] = 0.93246951
        x[1] = - x[0]
        x[2] = 0.66120939
        x[3] = - x[2]
        x[4] = 0.23861919
        x[5] = - x[4]
        c[0] = 0.17132449
        c[1] = c[0]
        c[2] = 0.36076157
        c[3] = c[2]
        c[4] = 0.46791393
        c[5] = c[4]
    if n == 7:
        x[0] = 0.94910791
        x[1] = - x[0]
        x[2] = 0.74153119
        x[3] = - x[2]
        x[4] = 0.40584515
        x[5] = - x[4]
         x[6] = 0.
        c[0] = 0.12948497
        c[1] = c[0]
        c[2] = 0.27970539
        c[3] = c[2]
        c[4] = 0.38183005
        c[5] = c[4]
        c[6] = 0.41795918
    if n == 8:
        x[0] = 0.96028986
        x[1] = - x[0]
        x[2] = 0.79666648
        x[3] = - x[2]
        x[4] = 0.52553241
        x[5] = - x[4]
        x[6] = 0.18343464
        x[7] = - x[6]
        c[0] = 0.10122854
        c[1] = c[0]
        c[2] = 0.22238103
        c[3] = c[2]
        c[4] = 0.31370665
        c[5] = c[4]
        c[6] = 0.36268378
        c[7] = c[6]
        
    c1 = (b + a)/2. 
    c2 = (b - a)/2.
    sum = 0.
    for i in range(n):
        sum = sum + c[i] * f(c1 + c2 * x[i]) 
    return c2 * sum
'''
    st.code(code, language='python')

    st.write(r"""
Решение задачи приближенного вычисления рассматриваемого интеграла да­ется следующей программой.
    """)
    st.subheader("Основной код")
    st.write(r"Приближенное вычисление интеграла на интервале $[a,b]$")
    code = '''
import math as mt
from gauss import gauss 
def f(x):
    return - mt.log(x) / (1. - x) 
a = 0.
b = 1.
for n in range(2, 9):
    I = gauss(f, a, b, n)
    print ('n =', n, 'Integral =', I)
Iexact = mt.pi ** 2 / 6.
print ('Exact value =', Iexact)
  '''
    st.code(code, language='python')

if chart_visual == 'Параметрические расчеты с NumPy':
    st.header('Параметрические расчеты с NumPy')

    st.write(r"""Интервал: $[a,b]$""")
    a, b = st.slider(' ', 0.0, 1.0, (0., 1.), 0.01)

    import numpy as np
    def gauss(f, a, b, n):
        '''
Integral of f(x) from a to b computed by 
Gauss-Legendre quadrature using m nodes.
        '''
        if n > 8 or n < 2:
            print ('The number of nodes must be greater than 2 and less than 8')
            st.write('The number of nodes must be greater than 2 and less than 8')
            return 0
        x = np.zeros(n, 'float')
        c = np.zeros(n, 'float')
        if n == 2:
            x[0] = 0.57735027
            x[1] = - x[0]
            c[0] = 1.
            c[1] = c[0]
        if n == 3:
            x[0] = 0.77459667
            x[1] = - x[0]
            x[2] = 0.
            c[0] = 0.55555556
            c[1] = c[0]
            c[2] = 0.88888889
        if n == 4:
            x[0] = 0.86113631
            x[1] = - x[0]
            x[2] = 0.33998104
            x[3] = - x[2]
            c[0] = 0.34785485 
            c[1] = c[0]
            c[2] = 0.65214515
            c[3] = c[2]
        if n == 5:
            x[0] = 0.90617985
            x[1] = - x[0]
            x[2] = 0.53846931
            x[3] = - x[2]
            x[4] = 0.
            c[0] = 0.23692689
            c[1] = c[0]
            c[2] = 0.47862867
            c[3] = c[2]
            c[4] = 0.56888889
        if n == 6:
            x[0] = 0.93246951
            x[1] = - x[0]
            x[2] = 0.66120939
            x[3] = - x[2]
            x[4] = 0.23861919
            x[5] = - x[4]
            c[0] = 0.17132449
            c[1] = c[0]
            c[2] = 0.36076157
            c[3] = c[2]
            c[4] = 0.46791393
            c[5] = c[4]
        if n == 7:
            x[0] = 0.94910791
            x[1] = - x[0]
            x[2] = 0.74153119
            x[3] = - x[2]
            x[4] = 0.40584515
            x[5] = - x[4]
            x[6] = 0.
            c[0] = 0.12948497
            c[1] = c[0]
            c[2] = 0.27970539
            c[3] = c[2]
            c[4] = 0.38183005
            c[5] = c[4]
            c[6] = 0.41795918
        if n == 8:
            x[0] = 0.96028986
            x[1] = - x[0]
            x[2] = 0.79666648
            x[3] = - x[2]
            x[4] = 0.52553241
            x[5] = - x[4]
            x[6] = 0.18343464
            x[7] = - x[6]
            c[0] = 0.10122854
            c[1] = c[0]
            c[2] = 0.22238103
            c[3] = c[2]
            c[4] = 0.31370665
            c[5] = c[4]
            c[6] = 0.36268378
            c[7] = c[6]
        
        c1 = (b + a)/2. 
        c2 = (b - a)/2.
        sum = 0.
        for i in range(n):
            sum = sum + c[i] * f(c1 + c2 * x[i]) 
        return c2 * sum
        
    import math as mt
    #from gauss import gauss 
    def f(x):
        return - mt.log(x) / (1. - x) 
    #a = 0.
    #b = 1.
    for n in range(2, 9):
        I = gauss(f, a, b, n)
        print ('n =', n, 'Integral =', I)
        st.write('n =', n, 'Integral =', I)
    Iexact = mt.pi ** 2 / 6.
    print ('Exact value =', Iexact)
    st.write('Exact value =', Iexact)

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
    
    st.write(r"""Интервал: $[a,b]$""")
    a, b = st.slider(' ', 0.0, 1.0, (0., 1.), 0.01)

    import numpy as np
    import math as mt
    import scipy as sp
    import scipy.integrate
    from scipy import integrate
    from scipy.special import legendre
    from scipy.special.orthogonal import p_roots
    from scipy.integrate import quad

    def f(x):
        return (- mt.log(x) / (1. - x))
    Iexact, err = integrate.quad(f, 0, 1)

    st.write('Exact value =', Iexact)

    I1 = scipy.integrate.quadrature(f, a, b)
    st.write('value =', I1)

    #from numpy import exp
    #f = lambda x:exp(-x**2)
    #i = scipy.integrate.quad(f, 0, 1)
    #print (i) 

    #I = scipy.integrate.fixed_quad(f, 0, 1, args=(), n=5)
    #I = scipy.integrate.quadrature(f, a, b)
    #print(I.val)

if chart_visual == 'Сравнение':
    st.header('Сравнение')
    st.write(r"""Интервал: $[a,b]$""")
    a, b = st.slider(' ', 0.0, 1.0, (0., 1.), 0.01)

    import numpy as np
    def gauss(f, a, b, n):
        '''
Integral of f(x) from a to b computed by 
Gauss-Legendre quadrature using m nodes.
        '''
        if n > 8 or n < 2:
            print ('The number of nodes must be greater than 2 and less than 8')
            st.write('The number of nodes must be greater than 2 and less than 8')
            return 0
        x = np.zeros(n, 'float')
        c = np.zeros(n, 'float')
        if n == 2:
            x[0] = 0.57735027
            x[1] = - x[0]
            c[0] = 1.
            c[1] = c[0]
        if n == 3:
            x[0] = 0.77459667
            x[1] = - x[0]
            x[2] = 0.
            c[0] = 0.55555556
            c[1] = c[0]
            c[2] = 0.88888889
        if n == 4:
            x[0] = 0.86113631
            x[1] = - x[0]
            x[2] = 0.33998104
            x[3] = - x[2]
            c[0] = 0.34785485 
            c[1] = c[0]
            c[2] = 0.65214515
            c[3] = c[2]
        if n == 5:
            x[0] = 0.90617985
            x[1] = - x[0]
            x[2] = 0.53846931
            x[3] = - x[2]
            x[4] = 0.
            c[0] = 0.23692689
            c[1] = c[0]
            c[2] = 0.47862867
            c[3] = c[2]
            c[4] = 0.56888889
        if n == 6:
            x[0] = 0.93246951
            x[1] = - x[0]
            x[2] = 0.66120939
            x[3] = - x[2]
            x[4] = 0.23861919
            x[5] = - x[4]
            c[0] = 0.17132449
            c[1] = c[0]
            c[2] = 0.36076157
            c[3] = c[2]
            c[4] = 0.46791393
            c[5] = c[4]
        if n == 7:
            x[0] = 0.94910791
            x[1] = - x[0]
            x[2] = 0.74153119
            x[3] = - x[2]
            x[4] = 0.40584515
            x[5] = - x[4]
            x[6] = 0.
            c[0] = 0.12948497
            c[1] = c[0]
            c[2] = 0.27970539
            c[3] = c[2]
            c[4] = 0.38183005
            c[5] = c[4]
            c[6] = 0.41795918
        if n == 8:
            x[0] = 0.96028986
            x[1] = - x[0]
            x[2] = 0.79666648
            x[3] = - x[2]
            x[4] = 0.52553241
            x[5] = - x[4]
            x[6] = 0.18343464
            x[7] = - x[6]
            c[0] = 0.10122854
            c[1] = c[0]
            c[2] = 0.22238103
            c[3] = c[2]
            c[4] = 0.31370665
            c[5] = c[4]
            c[6] = 0.36268378
            c[7] = c[6]
        
        c1 = (b + a)/2. 
        c2 = (b - a)/2.
        sum = 0.
        for i in range(n):
            sum = sum + c[i] * f(c1 + c2 * x[i]) 
        return c2 * sum
        
    import math as mt
    import time 
    time_0 = time.time() 
    def f(x):
        return - mt.log(x) / (1. - x) 
    #a = 0.
    #b = 1.
    for n in range(2, 9):
        I = gauss(f, a, b, n)
        print ('n =', n, 'Integral =', I)
        st.write('n =', n, 'Integral =', I)
    Iexact = mt.pi ** 2 / 6.
    print ('Exact value =', Iexact)
    st.write('Exact value =', Iexact)
    time_numpy = time.time() - time_0
    st.write ('Время NumPy=', time_numpy)

  # Мой код
 
    time_1 = time.time()

    time_scipy = time.time() - time_1
    st.write ('Время SciPy=', time_scipy)


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
