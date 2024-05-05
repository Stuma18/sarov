import streamlit as st
import math as mt

chart_visual = st.sidebar.radio('Содержание', 
    ('Главная', 'Постановка задачи', 'Описание алгоритма', 'Код с NumPy', 'Параметрические расчеты с NumPy', 'Код со SciPy', 'Параметрические расчеты со SciPy', 'Сравнение', 'Спасибо'))

if chart_visual == 'Главная':
  st.header("Задачи минимизации функций")
  st.header("Задание 9.1")
  st.subheader("Подготовила студентка первого курса магистратуры, группы СТФИ-122")
  st.subheader("Студеникина Мария")

if chart_visual == 'Постановка задачи':
  st.header('Постановка задачи')
  st.write(r"""
Напишите программу для нахождения минимума функции
одной переменной $f(x)$ на интервале $[a,b]$ методом золотого сечения.

С ее помощью найдите минимум функции

$\begin{aligned}
  (x^2 - 6 x +12) (x^2 + 6 x +12)^{-1}
\end{aligned}$

на интервале $[0,20]$.

Решите также эту задачу с помощью библиотеки SciPy.
""")

if chart_visual == 'Описание алгоритма':
    st.header('Описание алгоритма')
    st.write(r"""
Точка $x^{1}$ является точкой золотого сечения, если
""")
    st.write(r"""
$$
\frac{b - a}{x^{1} - a} = \frac{x^{1} - a}{b - x^{1}}
$$
""")

    st.write(r"""
Исходя из этого определения имеем
""")
    st.write(r"""
$$  
    x^{1}= a + \frac{\sqrt{5} -1}{2}(b-a)
$$ 
""")

    st.write(r"""
Аналогично для $x^{2}$
""")
    st.write(r"""
$$  
    \frac{x^{1}-a}{x^{2}-a}=\frac{x^{2}-a}{x^{1}-x^{2}}
$$  
""")
    st.write(r"""
и поэтому
""")
    st.write(r"""
$$
    x^{2}= a + \frac{3-\sqrt{5}}{2}(b-a)
$$ 
""")
    st.write(r"""
далее проводится сравнение значений функции в четырех точках $a$, $x^{2}$, $x^{1}$, $b$
$$  
    x^{1}= a + \frac{\sqrt{5} -1}{2}(b-a)
$$ 
""")
    st.write(r"""
На основе сравнения значений функции $f(x)$ в этих точках проводится 
итерационное уточнение интервала, на котором функция имеет минимум.
Число итераций $n$ для достижения необходимой точности
$\epsilon$ 
в определении точки минимума определяется равенством
""")
    st.write(r"""
$$
    \mid b-a\mid c^{n} = \epsilon
$$  
""")
    st.write(r"""
$$
    c=\frac{\sqrt{5}-1}{2}
$$
""")

if chart_visual == 'Код с NumPy':
    st.header('Код с NumPy')
    st.write(r"""
В модуле golden функция $golden()$ обеспечивает нахождение минимума 
функции одной переменной $f(x)$ на интервале [$a$, $b$].
""")
    st.subheader("Модуль golden")
    code = '''
import math as mt
def golden (f, a, b, tol = 1.0e-10):
    c1 = (mt.sqrt(5) - 1) / 2
    c2 = 1 - c1
    nIt = int(mt.ceil(mt.log(tol / abs(b - a)) / mt.log(c1)))
    # Первый шаг
    x1 = c1 * a + c2 * b
    x2 = c2 * a + c1 * b
    f1 = f(x1)
    f2 = f(x2)
    # Итерация
    for i in range(nIt):
        if f1 > f2: 
            a = x1
            x1 = x2
            f1 = f2
            x2 = c2 * a + c1 * b
            f2 = f(x2)
        else:
            b = x2
            x2 = x1
            f2 = f1
            x1 = c1 * a + c2 * b
            f1 = f(x1)
        
    if f1 < f2: 
        return x1, f1
    else:
        return x2,f2
    '''
    st.code(code, language='python')

    st.write(r'''
Для оценки интервала, на котором функция $f(x)$ достигает минимума, 
будем использовать график $y=f(x)$. С учетом этого для нахождения 
минимума функции

$\begin{aligned}
  (x^2 - 6 x +12) (x^2 + 6 x +12)^{-1}
\end{aligned}$

используется следующая программа.
''')
    
    code = '''
import numpy as np
import matplotlib.pyplot as plt
from golden import golden
def f(x):
    return (x ** 2 - 6 * x + 12) / (x ** 2 + 6 * x + 12)
a = 0
b = 20
x = np.linspace(a, b, 200)
y = f(x)
plt.plot(x, y)
plt.xlabel('x')
plt.grid(True)
plt.show()
    
xMin, fMin = golden(f, a, b)
print ('xMin =', xMin)
print ('fMin =', fMin)
    '''
    st.code(code, language='python')

if chart_visual == 'Параметрические расчеты с NumPy':
    st.header('Параметрические расчеты с NumPy')

    select_tol = st.selectbox('Точность', ('1.0e-5', '1.0e-6', '1.0e-7', '1.0e-8', '1.0e-9', '1.0e-10', '1.0e-11', '1.0e-12', '1.0e-13', '1.0e-14', '1.0e-15'))
    if select_tol == '1.0e-5':
        tol1 = 1.0e-5
    if select_tol == '1.0e-6':
        tol1 = 1.0e-6
    if select_tol == '1.0e-7':
        tol1 = 1.0e-7
    if select_tol == '1.0e-8':
        tol1 = 1.0e-8
    if select_tol == '1.0e-9':
        tol1 = 1.0e-9
    if select_tol == '1.0e-10':
        tol1 = 1.0e-10
    if select_tol == '1.0e-11':
        tol1 = 1.0e-11
    if select_tol == '1.0e-12':
        tol1 = 1.0e-12
    if select_tol == '1.0e-13':
        tol1 = 1.0e-13
    if select_tol == '1.0e-14':
        tol1 = 1.0e-14
    if select_tol == '1.0e-15':
        tol1 = 1.0e-15
    
    nIt = st.slider('Количество итераций', 1, 100, 31)
    st.write("Количество итераций = ", nIt)
    
    import math as mt
    def golden (f, a, b, tol = tol1):
        c1 = (mt.sqrt(5) - 1) / 2
        c2 = 1 - c1
        #nIt = int(mt.ceil(mt.log(tol / abs(b - a)) / mt.log(c1)))
        print ('nIt =', nIt)
        x1 = c1 * a + c2 * b
        x2 = c2 * a + c1 * b
        f1 = f(x1)
        f2 = f(x2)
        for i in range(nIt):
            if f1 > f2: 
                a = x1
                x1 = x2
                f1 = f2
                x2 = c2 * a + c1 * b
                f2 = f(x2)
            else:
                b = x2
                x2 = x1
                f2 = f1
                x1 = c1 * a + c2 * b
                f1 = f(x1)
            
#            i += i
        
        if f1 < f2: 
            return x1, f1
        else:
            return x2,f2
    
    import numpy as np
    import matplotlib.pyplot as plt
    #from golden import golden
    def f(x):
        return (x ** 2 - 6 * x + 12) / (x ** 2 + 6 * x + 12)
    a = 0
    b = 20
    x = np.linspace(a, b, 200)
    y = f(x)
    plt.plot(x, y)
    plt.xlabel('x')
    plt.grid(True)
    plt.show()

    xMin, fMin = golden(f, a, b)

    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    #plt.grid(True)
    st.pyplot(fig)
    
#    xMin, fMin = golden(f, a, b)
    print ('xMin =', xMin)
    print ('fMin =', fMin)
    st.write ('xMin =', xMin)
    st.write ('fMin =', fMin)
#    st.write ('fMin =', i)

if chart_visual == 'Код со SciPy':
    st.header('Код со SciPy')
    st.write(r"""
scipy.optimize.golden(func, args = (), brack = None, tol = 1.4901161193847656e-08,
full_output = 0, maxiter = 5000)

$\textbf{Параметры:}$ 

$\bullet$ func:

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ Целевая функция для минимизации


$\bullet$ argstuple:

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ Дополнительные аргументы (если они присутствуют), передаваемые в func

$\bullet$ brack:

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ Тройной (a,b,c), где (a < b <c) и func(b) < func(a),func(c).

$\bullet$ tol:

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ x критерий остановки допуска

$\bullet$ full_output:

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ Если True, возврат необязательных выходных данных

$\bullet$ maxiter:

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ Максимальное количество итераций для выполнения

Библиотечная функция scipy.optimize.golden возвращает точку минимума функции 
одной переменной с помощью метода золотого сечения.
""")
    code = '''
from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np

a = 0
b = 20

def f(x):
    return (x ** 2 - 6 * x + 12) / (x ** 2 + 6 * x + 12)
    
minimum = optimize.golden(f, brack = (a, b))
y = (minimum ** 2 - 6 * minimum + 12) / (minimum ** 2 + 6 * minimum + 12)
    
# для графика
x = np.linspace(a, b, 200)
y1 = f(x)
fig, ax = plt.subplots()
ax.plot(x, y1)
plt.xlabel('x')
st.pyplot(fig)

print ('xMin =', minimum)
print ('fMin =', y)
    '''
    st.code(code, language='python')

if chart_visual == 'Параметрические расчеты со SciPy':
    st.header('Параметрические расчеты со SciPy')


    select_tol = st.selectbox('Точность', ('1.0e-5', '1.0e-6', '1.0e-7', '1.0e-8', '1.0e-9', '1.0e-10', '1.0e-11', '1.0e-12', '1.0e-13', '1.0e-14', '1.0e-15'))
    if select_tol == '1.0e-5':
        tol1 = 1.0e-5
    if select_tol == '1.0e-6':
        tol1 = 1.0e-6
    if select_tol == '1.0e-7':
        tol1 = 1.0e-7
    if select_tol == '1.0e-8':
        tol1 = 1.0e-8
    if select_tol == '1.0e-9':
        tol1 = 1.0e-9
    if select_tol == '1.0e-10':
        tol1 = 1.0e-10
    if select_tol == '1.0e-11':
        tol1 = 1.0e-11
    if select_tol == '1.0e-12':
        tol1 = 1.0e-12
    if select_tol == '1.0e-13':
        tol1 = 1.0e-13
    if select_tol == '1.0e-14':
        tol1 = 1.0e-14
    if select_tol == '1.0e-15':
        tol1 = 1.0e-15
    
    nIt = st.slider('Количество итераций', 1, 100, 31)
    st.write("Количество итераций = ", nIt)

    from scipy import optimize
    import matplotlib.pyplot as plt
    import numpy as np

    a = 0
    b = 20

    def f(x):
        return (x ** 2 - 6 * x + 12) / (x ** 2 + 6 * x + 12)
    
    minimum = optimize.golden(f, brack = (a, b), tol=tol1, maxiter=nIt)
    y = f(minimum)
    
    # для графика
    x = np.linspace(a, b, 200)
    y1 = f(x)
    fig, ax = plt.subplots()
    ax.plot(x, y1)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    #plt.grid(True)
    st.pyplot(fig)

    print ('xMin =', minimum)
    print ('fMin =', y)

    st.write ('xMin =', minimum)
    st.write ('fMin =', y)

if chart_visual == 'Сравнение':
    st.header('Сравнение')

    import math as mt
    import time 
    t0 = time.time()
    def golden (f, a, b, tol = 1.0e-10):
        c1 = (mt.sqrt(5) - 1) / 2
        c2 = 1 - c1
        nIt = int(mt.ceil(mt.log(tol / abs(b - a)) / mt.log(c1)))
        x1 = c1 * a + c2 * b
        x2 = c2 * a + c1 * b
        f1 = f(x1)
        f2 = f(x2)
        for i in range(nIt):
            if f1 > f2: 
                a = x1
                x1 = x2
                f1 = f2
                x2 = c2 * a + c1 * b
                f2 = f(x2)
            else:
                b = x2
                x2 = x1
                f2 = f1
                x1 = c1 * a + c2 * b
                f1 = f(x1)
        
        if f1 < f2: 
            return x1, f1
        else:
            return x2,f2
    import numpy as np
    def f(x):
        return (x ** 2 - 6 * x + 12) / (x ** 2 + 6 * x + 12)
    a = 0
    b = 20
    #x = np.linspace(a, b, 200)
    #y = f(x)
    
    xMin, fMin = golden(f, a, b)
    t_numpy = time.time() - t0
    print ('xMin =', xMin)
    print ('fMin =', fMin)



    from scipy import optimize
    import time 
    t1 = time.time()
    minimum = optimize.golden(f, brack = (a, b))
    y = (minimum ** 2 - 6 * minimum + 12) / (minimum ** 2 + 6 * minimum + 12)

    t_scipy = time.time() - t1
    print ('xMin =', minimum)
    print ('fMin =', y)

    xMinW = 3.46410161513775
    yMinW = (xMinW ** 2 - 6 * xMinW + 12) / (xMinW ** 2 + 6 * xMinW + 12)

    #st.write(t_numpy)
    #st.write(t_scipy)
    #st.write(t_scipy - t_numpy)


    #st.write(r"""
#$$
#\begin{Vmatrix*}[r]
#Решение & NumPy & SciPy & WolframAlpha \\ 
#xMin & 3.46410163303 & 3.4641016339546207 & 3.46410161513775\\ 
#fMin & 0.0717967697245 & 0.0717967697244908 & 0.07179676972449082
#\end{Vmatrix*}
#$$
#""")

    st.subheader("Полученные ответы")

    st.write(r"""
$$
\def\arraystretch{1.5}
    \begin{array}
{c : c: c : c}
    Решение & NumPy & SciPy & WolframAlpha \\
\hline
    xMin & 3.46410163303 & 3.4641016339546207 & 3.46410161513775 \\
    \hdashline
    fMin & 0.0717967697245 & 0.0717967697244908 & 0.07179676972449082
\end{array}

$$
""")

#    st.write(- minimum + xMinW)

#    st.write(r"""
#$$
#\begin{Vmatrix*}[r]
#Сравнение & WolframAlpha - NumPy & WolframAlpha - SciPy\\ 
#xMin & -1.7887608727562565e-08 & -1.881687072824434e-08\\ 
#fMin & 5.551115123125783e-17 & 2.7755575615628914e-17
#\end{Vmatrix*}
#$$
#""")

    st.subheader("Сравнение ответов")

    st.write(r"""
$$
\def\arraystretch{1.5}
    \begin{array}
{c : c: c : c}
    Сравнение & WolframAlpha - NumPy & WolframAlpha - SciPy\\
\hline
    xMin & -1.7887608727562565e-08 & -1.881687072824434e-08\ \\
    \hdashline
    fMin & 5.551115123125783e-17 & 2.7755575615628914e-17
\end{array}

$$
""")

#    st.write(r"""
#$$
#\begin{Vmatrix*}[r]
#Сравнение & Время \\ 
#Время NumPy & 0.00010609626770019531 \\
#Время SciPy & 0.000225067138671875 \\
#Время SciPy - NumPy & 5.2928924560546875e-05 \\
#\end{Vmatrix*}
#$$
#""")

    st.subheader("Сравнение времени")

    st.write(r"""
$$
\def\arraystretch{1.5}
    \begin{array}
{c : c:}
    Сравнение & Время\\
\hline
    Время NumPy & 0.00010609626770019531 \\
    \hdashline
    Время SciPy & 0.000225067138671875 \\
    \hdashline
    Время SciPy - NumPy & 5.2928924560546875e-05
\end{array}

$$
""")
  
if chart_visual == 'Спасибо':
    from PIL import Image
    image = Image.open('9.1.jpg')
    st.image(image)