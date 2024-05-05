import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time 

chart_visual = st.sidebar.radio('Содержание', 
    ('Главная', 'Постановка задачи', 'Описание алгоритма', 'Код с NumPy', 'Параметрические расчеты с NumPy', 'Код со SciPy', 'Параметрические расчеты со SciPy', 'Сравнение', 'Спасибо'))

if chart_visual == 'Главная':
    st.header("Задача Коши для дифференциальных уравнений")
    st.header("Задание 13.2")
    st.subheader("Подготовила студентка первого курса магистратуры, группы СТФИ-122")
    st.subheader("Студеникина Мария")

if chart_visual == 'Постановка задачи':
    st.header('Постановка задачи')
    st.write(r"""Напишите программу для численного решения задачи Коши для
системы обыкновенных дифференциальных уравнений
с использованием двухслойной схемы с весом при решении
системы нелинейных уравнений на новом временном слое методом Ньютона.
Используйте эту программу для решения задачи Коши
(модель Лотка-Вольтерра)

$\begin{aligned}
  \frac{d y_1}{dt} = y_1 - y_1 y_2,
  \quad \frac{d y_2}{dt} = - y_2 + y_1 y_2,
  \quad 0 < t \leq 10,
\end{aligned}$

$\begin{aligned}
  y_1(0) = 2,
  \quad y_2(0) = 2 .
\end{aligned}$

Решите также эту задачу с помощью библиотеки SciPy.
""")

if chart_visual == 'Описание алгоритма':
    st.header('Описание алгоритма')

    st.subheader('Двухслойная схема с весом')
    st.write(r"""
$$
    \frac{du}{dt}= f(t,u) \: \: \: \: \: \: \: \: \: \:  u = (u_1, u_2)
$$
$$
    \frac{u^{n+1}-u^{n}}{\tau} = \sigma f(t^{n},u^{n}) + (1-\sigma) f(t^{n+1},u^{n+1})
$$
$$
    \sigma \in [0,1]
$$
$$
    u^{n+1}-(1-\sigma)\tau f(t^{n+1},u^{n+1})= u^{n}+ \tau \sigma f(t^{n},u^{n+1})
$$
$$
    F(u^{n+1}) = \varphi^{n}
$$
""")
    st.subheader('Метод Ньютона')
    st.write(r"""
$$
    F(u_{k+1}) \approx \varphi
$$

где $k$ - номер иттерации

$$
    F(u_{k}) + \tau F(u_{k}) \approx F(u_{k+1})
$$

$F$ - Якоби
""")

if chart_visual == 'Код с NumPy':
    st.header('Код с NumPy')
    st.write(r"""
Реализация двухслойной (одношаговой) схемы с весом проводится функцией oneStep() 
в модуле oneStep. Для решения системы нелинейных уравнений используется модуль newton.
""")
    st.subheader("Матрица Якоби")
    code = '''
import numpy as np
def jacobian(f, x):
    h = 1.0e-4
    n = len(x)
    Jac = np.zeros([n,n])
    f0 = f(x)
    for i in np.arange(0,n,1):
        tt = x[i]
        x[i] = tt + h
        f1= f(x)
        x[i] = tt
        Jac [:,i] = (f1 - f0)/h
    return Jac, f0
'''
    st.code(code, language='python')

    st.subheader("Метод Ньютона")
    code = '''
import numpy as np
from jacobian import jacobian
def newton(f, x, tol = 1.0e-9):
    iterMax = 50
    for i in range(iterMax):
        Jac, fO = jacobian(f, x)
        if np.sqrt(np.dot(fO, fO) / len(x)) < tol:
            return x, i                 
        dx = np.linalg.solve(Jac, fO)
        x = x - dx
        print ("Too many iterations for the Newton method")
'''
    st.code(code, language='python')

    st.subheader("Модуль oneStep")
    code = '''
import numpy as np
from newton import newton
def oneStep(f, t0, y0, tEnd, nTime, theta):
    """
    Решите задачу с начальным значением y' = f(t,y) 
    с помощью одноэтапных неявных методов.
    t0,y0 - начальные условия,
    tEnd - конечное значение t,
    time - количество шагов.
    """
    tau = (tEnd - t0) / nTime
    def f1(y1):
        f1 = y1 - tau * theta * f(t0 + tau, y1) - y0 - tau * (1.- theta) * f(t0 + tau, y0)
        return f1
    t = []
    y = []
    t.append(t0)
    y.append(y0)
    for i in range(nTime):
        r = y0 - tau * f(t0 + tau, y0)
        y1, iter = newton(f1, r)
        y0 = y1
        t0 = t0 + tau
        t.append(t0)
        y.append(y0)
    return np.array(t), np.array(y)
'''
    st.code(code, language='python')

    st.subheader("Основной код")
    st.write(r"Решение задачи при $\theta = 0.5$")
    code = '''
import numpy as np
import matplotlib.pyplot as plt 
from oneStep import oneStep

def f(t, y):
    f = np.zeros((2),'float')
    f[0] = y[0] - y[0] * y[1]
    f[1] = - y[1] + y[0] * y[1]
    return f 
t0 = 0.
tEnd = 10.
y0 = np.array([2., 2. ])
nTime = 50
theta = 0.5
t, y = oneStep(f, t0, y0, tEnd, nTime, theta)

# для графика
for n in range(0, 2):
    r = y[:, n]
    st1 = '$y_l$'
    sg = '-'
    if n == 1:
        st1 = '$у_2$'
        sg = '--'
    plt.plot(t, r, sg, label = st1)
plt.legend(loc = 0)
plt.xlabel('$t$')
plt.grid(True)
plt.show()
  '''
    st.code(code, language='python')

if chart_visual == 'Параметрические расчеты с NumPy':
    st.header('Параметрические расчеты с NumPy')
    
    st.write(r"""Интервал решения задачи:""")
    t_1 = st.slider(' ', -10, 20, (0, 10), 1)
    
#    st.write(r"""$\theta =$""")
#    theta_1 = st.slider('', 0.1, 3.0, 0.5)

    st.write(r"""Шаги по времени:""")
    nTime_1 = st.slider('', 3, 150, 50)

    import numpy as np
    import time 
    t0 = time.time()
    def jacobian(f, x):
        h = 1.0e-4
        n = len(x)
        Jac = np.zeros([n,n])
        f0 = f(x)
        for i in np.arange(0,n,1):
            tt = x[i]
            x[i] = tt + h
            f1= f(x)
            x[i] = tt
            Jac [:,i] = (f1 - f0)/h
        return Jac, f0

    def newton(f, x, tol = 1.0e-9):
        iterMax = 50
        for i in range(iterMax):
            Jac, fO = jacobian(f, x)
            if np.sqrt(np.dot(fO, fO) / len(x)) < tol:
                return x, i                 
            dx = np.linalg.solve(Jac, fO)
            x = x - dx
            print ("Too many iterations for the Newton method")
    
    #import numpy as np
    #from newton import newton
    def oneStep(f, t0, y0, tEnd, nTime, theta):
        """
        Решите задачу с начальным значением y' = f(t,y) 
        с помощью одноэтапных неявных методов.
        t0,y0 - начальные условия,
        tEnd - конечное значение t,
        ntime - количество шагов.
        """
        tau = (tEnd - t0) / nTime
        def f1(y1):
            f1 = y1 - tau * theta * f(t0 + tau, y1) - y0 - tau * (1.- theta) * f(t0 + tau, y0)
            return f1
        t = []
        y = []
        t.append(t0)
        y.append(y0)
        for i in range(nTime):
            r = y0 - tau * f(t0 + tau, y0)
            y1, iter = newton(f1, r)
            y0 = y1
            t0 = t0 + tau
            t.append(t0)
            y.append(y0)
        return np.array(t), np.array(y)
    
    import numpy as np
    import matplotlib.pyplot as plt 
    #from oneStep import oneStep
    import time 
    time_0 = time.time()

    def f(t, y):
        f = np.zeros((2),'float')
        f[0] = y[0] - y[0] * y[1]
        f[1] = - y[1] + y[0] * y[1]
        return f 
    t0 = int(t_1[0])
    tEnd = int(t_1[1])
    y0 = np.array([2., 2. ])
    nTime = nTime_1
    theta = 0.5

    t_numpy, y_numpy = oneStep(f, t0, y0, tEnd, nTime, theta)

    fig, ax = plt.subplots()

    for n in range(0, 2):
        r = y_numpy[:, n]
        st1 = '$y_l$'
        sg = '-'
        if n == 1:
            st1 = '$у_2$'
            sg = '--'

        ax.plot(t_numpy, r, sg, label = st1)
    ax.legend(loc = 0)
    ax.set_xlabel('$t$')
    ax.grid()
    st.pyplot(fig)
    
    time_numpy = time.time() - time_0
    st.write ('Время =', time_numpy)


    t_ol = np.transpose(t_numpy)
    y_new = np.transpose(y_numpy)
  
    mat = np.vstack((t_ol, y_new))
    mat1 = np.transpose(mat)
    df = pd.DataFrame(
        mat1,
        columns=('t', 'y1 (NumPy)', 'y2 (NumPy)'))  
  
    st.table(df)

if chart_visual == 'Код со SciPy':
    st.header('Код со SciPy')

    st.subheader('scipy.integrate.solve_ivp')
    st.write(r"""

scipy.integrate.solve_ivp(fun, t_span, y0, method='RK45', t_eval=None, 
dense_output=False, events=None, vectorized=False, args=None, **options)


Эта функция численно интегрирует систему обыкновенных дифференциальных уравнений, 
заданных начальным значением.

##### Параметры

- fun - функция.

- t_span - интервал интегрирования.

- y0 - начальное условие.

- method - используемый метод интегрирования, 'RK45' (по умолчанию).

- t_eval - времена, когда нужно сохранить вычисленное решение, должны быть отсортированы и лежать в пределах t_span. Если None (по умолчанию), используйте точки, выбранные решателем.

- dense_output - следует ли вычислять непрерывное решение. Значение по умолчанию равно False.

- events - события для отслеживания. Если нет (по умолчанию), то никакие события отслеживаться не будут. Каждое событие происходит при нулях непрерывной функции времени и состояния. Каждая функция должна иметь событие подписи(t, y) и возвращать значение с плавающей точкой. 

- vectorized - pеализована ли функция векторизованным способом. По умолчанию — Ложь.

- args - дополнительные аргументы для передачи пользовательским функциям. Если заданы, дополнительные аргументы передаются всем пользовательским функциям.

**options

- first_step - Начальный размер шага. Значение по умолчанию равно None, это означает, что алгоритм сам выберет.

- max_step - Максимально допустимый размер шага. Значение по умолчанию равно np.inf, т.е. размер шага не ограничен и определяется исключительно алгоритмом.

##### Возвращается

- t - временные точки.

- y - значения решения при t.
    
    """)

    st.write(r"Решение задачи при $\theta = 0.5$")
    code = '''
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

def f(t, y):
    f = np.zeros((2),'float')
    f[0] = y[0] - y[0] * y[1]
    f[1] = - y[1] + y[0] * y[1]
    return f
t0 = 0.
tEnd = 10.
t1 = np.array([t0, tEnd])
y0 = np.array([2., 2. ])
nTime = 50
theta = 0.5
tau = ((tEnd - t0) / nTime)

sol = solve_ivp(f, t1, y0, first_step = tau, max_step = tau, tol = theta)

t = sol.t
y = np.transpose(sol.y)

# для графика
for n in range(0, 2):
    r = y[:, n]
    st1 = '$y_l$' 
    sg = "-"
    if n == 1:
        st1 = '$у_2$'
        sg = "--"
  
    plt.plot(t, r, sg, label = st1)
plt.legend(loc = 0)
plt.xlabel('$t$')
plt.grid(True)
plt.show()
    '''
    st.code(code, language='python')

if chart_visual == 'Параметрические расчеты со SciPy':
    st.header('Параметрические расчеты со SciPy')

    st.write(r"""Интервал решения задачи:""")
    t_1 = st.slider(' ', -10, 20, (0, 10), 1)
    
#    st.write(r"""$\theta =$""")
#    theta_1 = st.slider('', 0.1, 3.0, 0.5)

    st.write(r"""Шаги по времени:""")
    nTime_1 = st.slider('', 3, 150, 50)

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.integrate import solve_ivp
    import time 
    time_1 = time.time()

    def f(t, y):
        #f = np.zeros((2),'float')
        f = np.zeros((2))
        f[0] = y[0] - y[0] * y[1]
        f[1] = - y[1] + y[0] * y[1]
        return f
    t0 = int(t_1[0])
    tEnd = int(t_1[1])
    t1 = np.array([t0, tEnd])
    y0 = np.array([2., 2. ])
    nTime = nTime_1
    #theta = theta_1
    theta = 0.5
    tau = ((tEnd - t0) / nTime)

    sol = solve_ivp(f, t1, y0, first_step = tau, max_step = tau)
#    sol = solve_ivp(f, t1, y0, first_step = tau, max_step = tau)

    t = sol.t
    t = t[:nTime + 1]
    y = np.transpose(sol.y)
    y = y[:nTime + 1]

    fig, ax = plt.subplots()
    for n in range(0, 2):
        r = y[:, n]
        st1 = '$y_l$' 
        sg = "-"
        if n == 1:
            st1 = '$у_2$'
            sg = "--"

        ax.plot(t, r, sg, label = st1)
    ax.legend(loc = 0)
    ax.set_xlabel('$t$')
    ax.grid()
    st.pyplot(fig)
    
    t_scipy = sol.t
    t_scipy = t_scipy[:nTime + 1]
    y_scipy = np.transpose(sol.y)
    y_scipy = y_scipy[:nTime + 1]
  
    time_scipy = time.time() - time_1
    st.write ('Время SciPy = ', time_scipy)

    y_scipy = np.transpose(y_scipy )
  
    mat = np.vstack((t_scipy, y_scipy))
    mat1 = np.transpose(mat)
    df = pd.DataFrame(
        mat1,
        columns=('t', 'y1 (SciPy)', 'y2 (SciPy)'))  
  
    st.table(df)

if chart_visual == 'Сравнение':
    st.header('Сравнение')

    st.write(r"""Интервал решения задачи:""")
    t_1 = st.slider(' ', -10, 20, (0, 10), 1)
    
    st.write(r"""Шаги по времени:""")
    nTime_1 = st.slider('', 3, 150, 50)

    import numpy as np
    from scipy.integrate import solve_ivp
    import time 
    t0 = time.time()
    def jacobian(f, x):
        h = 1.0e-4
        n = len(x)
        Jac = np.zeros([n,n])
        f0 = f(x)
        for i in np.arange(0,n,1):
            tt = x[i]
            x[i] = tt + h
            f1= f(x)
            x[i] = tt
            Jac [:,i] = (f1 - f0)/h
        return Jac, f0

    def newton(f, x, tol = 1.0e-9):
        iterMax = 50
        for i in range(iterMax):
            Jac, fO = jacobian(f, x)
            if np.sqrt(np.dot(fO, fO) / len(x)) < tol:
                return x, i                 
            dx = np.linalg.solve(Jac, fO)
            x = x - dx
            print ("Too many iterations for the Newton method")

    def oneStep(f, t0, y0, tEnd, nTime, theta):
        tau = (tEnd - t0) / nTime
        def f1(y1):
            f1 = y1 - tau * theta * f(t0 + tau, y1) - y0 - tau * (1.- theta) * f(t0 + tau, y0)
            return f1
        t = []
        y = []
        t.append(t0)
        y.append(y0)
        for i in range(nTime):
            r = y0 - tau * f(t0 + tau, y0)
            y1, iter = newton(f1, r)
            y0 = y1
            t0 = t0 + tau
            t.append(t0)
            y.append(y0)
        return np.array(t), np.array(y)
    
    time_0 = time.time()

    def f(t, y):
        f = np.zeros((2),'float')
        f[0] = y[0] - y[0] * y[1]
        f[1] = - y[1] + y[0] * y[1]
        return f 
    t0 = int(t_1[0])
    tEnd = int(t_1[1])
    y0 = np.array([2., 2. ])
    nTime = nTime_1
    theta = 0.5
    t_numpy, y_numpy = oneStep(f, t0, y0, tEnd, nTime, theta)
    
    time_numpy = time.time() - time_0
    st.write ('Время NumPy = ', time_numpy)

    # мой код
    time_1 = time.time()
    t1 = np.array([t0, tEnd])
    tau = ((tEnd - t0) / nTime)

    sol = solve_ivp(f, t1, y0, first_step = tau, max_step = tau)


    t_scipy = sol.t
    t_scipy = t_scipy[:nTime + 1]
    y_scipy = np.transpose(sol.y)
    y_scipy = y_scipy[:nTime + 1]
  
    time_scipy = time.time() - time_1
    st.write ('Время SciPy = ', time_scipy)

    t_ol = np.transpose(t_numpy)
    y_new = np.transpose(y_numpy)
    y_scipy = np.transpose(y_scipy )
    y_del = y_new - y_scipy
  
    mat = np.vstack((t_ol, y_new, y_scipy, y_del))
    mat1 = np.transpose(mat)
    df = pd.DataFrame(
        mat1,
        columns=('t', 'y1 (NumPy)', 'y2 (NumPy)', 'y1 (SciPy)', 'y2 (SciPy)', 'Сравнение y1', 'Сравнение y2'))  
  
    st.table(df)

if chart_visual == 'Спасибо':
    from PIL import Image
    image = Image.open('13.2.jpg')
    st.image(image)
