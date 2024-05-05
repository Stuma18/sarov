import streamlit as st
import math as mt

chart_visual = st.sidebar.radio('Содержание', 
    ('Главная', 'Постановка задачи', 'Описание алгоритма', 'Код с NumPy', 'Параметрические расчеты с NumPy', 'Код со SciPy', 'Параметрические расчеты со SciPy', 'Сравнение', 'Спасибо'))

if chart_visual == 'Главная':
    st.header("Задачи минимизации функций")
    st.header("Задание 9.2")
    st.subheader("Подготовила студентка первого курса магистратуры, группы СТФИ-122")
    st.subheader("Студеникина Мария")

if chart_visual == 'Постановка задачи':
    st.header('Постановка задачи')
    st.write(r"""
Напишите программу для нахождения минимума функции
нескольких переменной $f(x)$ градиентным методом
при выборе итерационных параметров из минимума функции
одной переменной, который находиться методом золотого сечения
(смотри задание 9.1).

Проиллюстрируйте работу программы при минимизации
функции

$\begin{aligned}
  10(x_2 - x_1^2)^2 + (1-x_1)^2 .
\end{aligned}$

Решите также эту задачу с помощью библиотеки SciPy.
""")

if chart_visual == 'Описание алгоритма':
    st.header('Описание алгоритма')
    st.write(r"""
Для функции многих переменных $f(x), \ x = \{x_1,x_2,\ldots,x_n\}$ 

определим вектор первых частных производных (градиент)
$$

  f'(x) = \left \{ \frac{\partial f}{\partial x_i}(x)
  \right \} \equiv
  \left \{ \frac{\partial f}{\partial x_1}(x),
  \frac{\partial f}{\partial x_2}(x), \ldots,
  \frac{\partial f}{\partial x_n}(x) \right \} 

$$
Матрица вторых частных производный (гессиан) в точке $x$ есть
$$
  f''(x) = \left \{
  \frac{\partial^2 f}{\partial x_i \partial x_j}(x) \right \} 
$$
""")

    st.subheader('Итерационные методы')
    st.write(r"""
Пусть
$$
  x^{k+1} = x^{k} + \alpha_k h^k,
  \quad k=0,1,\ldots
$$
$h^k$ $-$ вектор направления $(k+1)$-го шага минимизации

коэффициент $\alpha_k$ $-$ длина шага

Вектор $h$ задает направление убывания функции $f(x)$ в точке $x$,

если $f(x+\alpha h) < f(x)$ при достаточно малых $\alpha > 0$
""")

    st.subheader('Метод спуска')
    st.write(r"""


$\;$ $\bull$ $\;$ $h^k$ задает направление убывания функции $f(x)$ в точке  $x^k$

$\;$ $\bull$ $\;$ $\alpha_k > 0$ такое, что

$$  
f(x^{k+1}) < f(x^k)
$$
  
$$

$$

В градиентном методе $h^k = - f'(x^k)$:

$$
  x^{k+1} = x^{k} - \alpha_k f'(x^k),
  
  k=0,1,\ldots
$$
""")
    st.subheader('Итерационные параметры')
    st.write(r"""
В итерационном методе
$$
  x^{k+1} = x^{k} + \alpha_k h^k,
  \quad k=0,1,\ldots
$$
нужно задавать итерационные параметры $\alpha_k, \ k = 0,1,\ldots$ 

Общее условие
$$
  f(x^k + \alpha_k h^k) = \min_{\alpha \geq 0} f(x^k + \alpha
  h^k)
$$
Решается дополнительная одномерная задача минимизации


Процедура дробления шага

Параметр $\alpha$ уменьшается, например, в два раза,
до тех пор пока не будет выполнено неравенство
$$
  f(x^k + \alpha h^k) < f(x^k) 
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

    st.write(r"""
В модуле grad функция $grad()$ находиться минимум функции многих пере­менных 
при аналитическом задании ее градиента.
""")
    st.subheader("Модуль grad")
    code = '''
import numpy as np
import math as mt
from golden import golden
def grad(F, GradF, x, d = 0.5, tol = 1.0e-10):
    # Линейная функция вдоль h
    def f(a1):
        return F(x + a1 * h) 
    gr0 = - GradF(x)
    h = gr0.copy()
    F0 = F(x)
    itMax = 500
    for i in range(itMax):
        # Функция минимизации ID
        a1, fMin = golden(f, 0, d) 
        x = x + a1 * h
        F1 = F(x)
        gr1 = - GradF(x)
        if (mt.sqrt(np.dot(gr1, gr1)) <= tol) or (abs(F0 - F1) < tol):
            return x, i + 1 
        h = gr1
        gr0 = gr1.copy() 
        F0 = F1
        print ("Gradient method did not converge (500 iterations)")
'''
    st.code(code, language='python')

    st.write(r"""
Для оценки начального приближения используется график исследуемой функции 

$\begin{aligned}
  10(x_2 - x_1^2)^2 + (1-x_1)^2 .
\end{aligned}$

с использованием следующей программы.
""")

    code = '''
import numpy as np
import matplotlib.pyplot as plt 
from grad import grad
def F(x):
    return 10 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2 
def GradF(x):
    gr = np.zeros((2),'float')
    gr[0] = -40 * (x[1] - x[0] ** 2) * x[0] - 2 * (1 - x[0]) 
    gr[1] = 20 * (x[1] - x[0] ** 2)
    return gr
    
# график функции
x = np.linspace(-2, 2, 101) 
y = np.linspace(-1, 3, 101) 
X, Y = np.meshgrid(x, y)
z = F([X, Y])
v = np.linspace(0, 10, 21)   
plt.contourf(x, y, z, v)
plt.colorbar()
plt.show()

# минимум функции
x0 = np.array([0, 0.1])
xMin, nit = grad(F, GradF, x0)
print ('XMin:', xMin)
print ('Number of iterations = ', nit)
'''
    st.code(code, language='python')

if chart_visual == 'Параметрические расчеты с NumPy':
    st.header('Параметрические расчеты с NumPy')

    select_tol = st.selectbox('Точность', ('1.0e-5', '1.0e-6', '1.0e-7', '1.0e-8', '1.0e-9', '1.0e-10', '1.0e-11', '1.0e-12'))
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
    
    nIt1 = st.slider('Количество итераций', 1, 500, 95)
    st.write("Количество итераций = ", nIt1)

    import math as mt
    import time 
    t0 = time.time()
    def golden (f, a, b, tol = tol1):
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
    import math as mt
    #from golden import golden
    def grad(F, GradF, x, d = 0.5, tol = tol1):
        # Line function along h
        def f(a1):
            return F(x + a1 * h) 
        gr0 = - GradF(x)
        h = gr0.copy()
        F0 = F(x)
        itMax = nIt1
        for i in range(itMax):
            # Minimization ID function 
            a1, fMin = golden(f, 0, d) 
            x = x + a1 * h
            F1 = F(x)
            gr1 = - GradF(x)
            if (mt.sqrt(np.dot(gr1, gr1)) <= tol) or (abs(F0 - F1) < tol):
                return x, i+1 
            h = gr1
            gr0 = gr1.copy() 
            F0 = F1
        print ("Gradient method did not converge (500 iterations)")
        st.write("Градиентный метод не сходится")
    
    import numpy as np
    import matplotlib.pyplot as plt 
    #from grad import grad
    def F(x):
        return 10 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2 
    def GradF(x):
        gr = np.zeros((2),'float')
        gr[0] = -40 * (x[1] - x[0] ** 2) * x[0] - 2 * (1 - x[0]) 
        gr[1] = 20 * (x[1] - x[0] ** 2)
        return gr
    
    # graph of function
    x = np.linspace(-2, 2, 101) 
    y = np.linspace(-1, 3, 101) 
    # y = 10 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
    X, Y = np.meshgrid(x, y)
    z = F([X, Y])
    v = np.linspace(0, 10, 21)
    
    plt.contourf(x, y, z, v)
    plt.colorbar()
    plt.show()
    
    #fig, ax = plt.subplots()
    #ax.contourf(x, y, z, v)
    #st.pyplot(fig)
    
    # minimum function
    x0 = np.array([0, 0.1])
    xMin, nit = grad(F, GradF, x0)

    fig, ax = plt.subplots()
    ax.contourf(x, y, z, v)
    plt.plot(xMin, '*')
    st.pyplot(fig)

    t_numpy = time.time() - t0

    print ('XMin:', xMin)
    print ('Number of iterations = ', nit)

    st.write ('xMin =', xMin)
    st.write ('Number of iterations =', nit)
    st.write ('Time =', t_numpy)

    st.write(r"""
Функция имеет выраженный овражный характер, что обуславлива­ет достаточно медленную 
сходимость градиентного метода.
""")

if chart_visual == 'Код со SciPy':
    st.header('Код со SciPy')

    st.write(r"""
Функция minim из пакета scipy.optimize предоставляет общий интерфейс для решения задач 
условной и безусловной минимизации скалярных функций нескольких переменных.


scipy.optimize.minimize(fun, x0, args=(), method=None, jac=None, 
hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)

$\textbf{Параметры:}$ 

$\bullet$ func:

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ Целевая функция, которая должна быть сведена к минимуму

$\bullet$ x0:

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ Первоначальное предположение. Массив вещественных элементов размера (n), где n - количество независимых переменных

$\bullet$ args:

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ Дополнительные аргументы, передаваемые целевой функции и ее производным (функциям fun, jack и jess).

$\bullet$ method:

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ Тип решателя. Должен быть одним из

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ $\bullet$ 'Nelder-Mead'

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ $\bullet$ 'Powell' 

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ $\bullet$ 'CG'

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ $\bullet$ 'BFGS'

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ $\bullet$ 'Newton-CG'

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ $\bullet$ 'L-BFGS-B'

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ $\bullet$ 'TNC'

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ $\bullet$ 'COBYLA'

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ $\bullet$ 'SLSQP'

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ $\bullet$ 'trust-constr'

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ $\bullet$ 'dogleg'

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ $\bullet$ 'trust-ncg'

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ $\bullet$ 'trust-exact'

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ $\bullet$ 'trust-krylov'

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ $\bullet$ 'custom' - вызываемый объект

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ Если не указано, выбирается один из BFGS, L-BFGS-B, SLSQP, в зависимости от того, имеет ли проблема ограничения или границы

$\bullet$ jac:

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ Способ вычисления вектора градиента. Только для CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg, trust-krylov, trust-exact и trust-constr

$\bullet$ hess:

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ Способ вычисления матрицы Гессиана. Только для Newton-CG, dogleg, trust-ncg, trust-krylov, trust-exact и trust-construction

$\bullet$ hessp:

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ Гессиан целевой функции, умноженный на произвольный вектор p. Только для Newton-CG, trust-ncg, trust-krylov, trust-constr

$\bullet$ bounds:

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ Границы переменных для методов Нелдера-Мида, L-BFGS-B, TNC, SLSQP, Пауэлла и trust-constr

$\bullet$ constraints:

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ Определение ограничений. Только для COBYLA, SLSQP и trust-construction

$\bullet$ tol:

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ Допуск к прекращению. Когда указан tol, выбранный алгоритм минимизации устанавливает некоторый соответствующий допуск для конкретного решателя, равный tol

$\bullet$ options:

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ Список вариантов решения

$\bullet$ callback:

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ Вызывается после каждой итерации

$\textbf{Вывод:}$ 

$\bullet$  res: OptimizeResult

$\;$ $\;$ $\;$ $\;$ $\;$ $\;$ Результат оптимизации, представленный в виде объекта OptimizeResult. Важными атрибутами являются: x - массив решений, success - Логический флаг, указывающий, успешно ли завершен оптимизатор, и сообщение, описывающее причину завершения.


Библиотечная функция scipy.optimize.minimize возвращает минимум функции 
одной или нескольких переменных с помощью разных методов.
""")
    st.subheader("Алгоритм Бройдена-Флетчера-Голдфарба-Шанно (BFGS)")

    st.write(r"""
Для получения более быстрой сходимости к решению, процедура BFGS использует градиент 
целевой функции. Градиент может быть задан в виде функции или вычисляться с помощью 
разностей первого порядка. В любом случае, обычно метод BFGS требует меньше вызовов 
функций.
""")

    st.write(r"""
""")
    code = '''
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

a = 0
b = 20
def F(x):
    return 10 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

def GradF(x):
    gr = np.zeros((2),'float')
    gr[0] = - 40 * (x[1] - x[0] ** 2) * x[0] - 2 * (1 - x[0]) 
    gr[1] = 20 * (x[1] - x[0] ** 2)
    return gr

x0 = np.array([0, 0.1])
res = minimize(F, x0, method='BFGS', jac= GradF, options={'disp': True})

# график функции
x = np.linspace(-2, 2, 101) 
y = np.linspace(-1, 3, 101) 
X, Y = np.meshgrid(x, y)
z = F([X, Y])
v = np.linspace(0, 10, 21)
plt.contourf(x, y, z, v)
plt.colorbar()
plt.show()

print(res.x)
print(res.nit)

'''
    st.code(code, language='python')

if chart_visual == 'Параметрические расчеты со SciPy':
    st.header('Параметрические расчеты со SciPy')

    select_tol = st.selectbox('Точность', ('1.0e-5', '1.0e-6', '1.0e-7', '1.0e-8', '1.0e-9', '1.0e-10', '1.0e-11', '1.0e-12'))
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
    
    nIt = st.slider('Количество итераций', 1, 20, 13)
    st.write("Количество итераций = ", nIt)

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    import time 
    t1 = time.time()

    a = 0
    b = 20
    def F(x):
        return 10 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    def GradF(x):
        gr = np.zeros((2),'float')
        gr[0] = - 40 * (x[1] - x[0] ** 2) * x[0] - 2 * (1 - x[0]) 
        gr[1] = 20 * (x[1] - x[0] ** 2)
        return gr

    x0 = np.array([0, 0.1])
    res = minimize(F, x0, method='BFGS', jac= GradF, tol=tol1, options={'maxiter':nIt, 'disp': True})

    # graph of function
    x = np.linspace(-2, 2, 101) 
    y = np.linspace(-1, 3, 101) 
    # y = 10 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
    X, Y = np.meshgrid(x, y)
    z = F([X, Y])
    v = np.linspace(0, 10, 21)

    plt.contourf(x, y, z, v)
    plt.colorbar()
    plt.show()

    fig, ax = plt.subplots()
    ax.contourf(x, y, z, v)
    plt.plot(res.x, '*')
    st.pyplot(fig)

    t_scipy = time.time() - t1

    st.write ('xMin =', res.x)
    st.write ('Number of iterations =', res.nit)
    st.write ('Time =', t_scipy)

if chart_visual == 'Сравнение':
    st.header('Сравнение')

    st.subheader("Полученные ответы")

    x_0 = - 0.999953820632928 + 0.9999999667010717
    x_1 = - 0.9999030255325715 + 0.9999999531647978
    tm = - 0.20331692695617676 + 0.14175105094909668
    #st.write (x_0)
    #st.write (x_1)
    #st.write (tm)

    st.write(r"""
$$
\def\arraystretch{1.5}
    \begin{array}
{c : c: c : c}
    Решение & NumPy & SciPy & Сравнение \: (SciPy - NumPy) \\
\hline
    xMin0 & 0.999953820632928 & 0.9999999667010717 & 4.614606814368205e-05 \\
    \hdashline
    xMin1 & 0.9999030255325715 & 0.9999999531647978 & 9.692763222635126e-05 \\
    \hdashline
    Number \: of \: iterations & 372 & 13 & 359 \\
    \hdashline
    Time & 0.20331692695617676 & 0.14175105094909668 & SciPy \: быстрее \: на \: 0.06156587600708008\\ 
\end{array}

$$
""")

if chart_visual == 'Спасибо':
    from PIL import Image
    image = Image.open('9.2.jpg')
    st.image(image)