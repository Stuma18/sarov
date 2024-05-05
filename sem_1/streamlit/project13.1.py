import streamlit as st
import math as mt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time 

chart_visual = st.sidebar.radio('Содержание', 
    ('Главная', 'Постановка задачи', 'Описание алгоритма', 'Код с NumPy', 'Параметрические расчеты с NumPy', 'Код со SciPy', 'Параметрические расчеты со SciPy', 'Сравнение', 'Спасибо'))

if chart_visual == 'Главная':
  st.header("Задача Коши для дифференциальных уравнений")
  st.header("Задание 13.1")
  st.subheader("Подготовила студентка первого курса магистратуры, группы СТФИ-122")
  st.subheader("Студеникина Мария")

if chart_visual == 'Постановка задачи':
  st.header('Постановка задачи')
  st.write(r"""Напишите программу для численного решения задачи Коши для
системы обыкновенных дифференциальных уравнений
явным методом Рунге-Кутта четвертого порядка.
Продемонстрируйте работоспособность этой программы
при решении задачи Коши

$\begin{aligned}
  \frac{d^2 u }{dt^2} = - \sin(u),
  \quad 0 < t < 4 \pi ,
\end{aligned}$

$\begin{aligned}
  u(0) = 1,
  \quad \frac{d u}{dt} (0) = 0.
\end{aligned}$

Решите также эту задачу с помощью библиотеки SciPy.
""")

if chart_visual == 'Описание алгоритма':
  st.header('Описание алгоритма')

  st.subheader('Задача Коши')
  st.write(r"""
$$
  \frac{d u_{i}(t)}{dt} = f_{i} (t,u_1,u_2,\ldots,u_m),
  \quad i = 1,2,\ldots,m ,
  \quad t > 0
$$
$$
  u_{i}(0) = u_{i}^{0},
  \quad i = 1,2,\ldots,m 
$$

Равномерная сетка по переменной $t$ с шагом $\tau > 0$
$$
  \omega_{\tau} = \{ t_n = n \tau, \ n = 0,1,\dots \}
$$
 $y^n$ — приближенное решение в точке $t = t_n$ 

""")
  st.subheader('Методы Рунге-Кутта')
  st.write(r"""
Метод записывается в общем виде
$$
  \frac{y^{n+1}-y^n}{\tau} = \sum_{i=1}^{s} b_i k_i
$$
$$
  k_i = f(t_n + c_i\tau, y^n + \tau \sum_{j=1}^{s} a_{ij} k_j),
  \quad i = 1,2,\ldots,s 
$$
Формула основана на $s$ вычислениях функции $f$ — $s$-стадийный метод 

$a_{ij} = 0$ при $j\ge i$ —
явный метод Рунге - Кутта
""")
  st.subheader('Явный метод Рунге-Кутта четвертого порядка')
  st.write(r"""

Одним из наиболее распространенных является явный метод
$$
  k_1 = f(t_n,y^{n}),
  \quad k_2 = f\bigg(t_n + \frac{\tau}{2}, y^{n} + \tau \frac{k_1}{2}\bigg)
$$
$$
  k_3 =f\bigg(t_n + \frac{\tau}{2}, y^{n} + \tau \frac{k_2}{2}\bigg),
  \quad k_4 = f(t_n + \tau, y^{n} + \tau k_3)
$$
$$
  \frac{y^{n+1}-y^n}{\tau} = \frac{1}{6} (k_1 + 2k_2 + 2k_3 + k_4)
$$

Компактная запись 

$$
\def\arraystretch{1.5}
    \begin{array}
{c | c}
    c & A \\
\hline
     & b^* \\
\end{array}

=
\def\arraystretch{1.5}
    \begin{array}
{c|cccc}
  0 & 0 & 0 & 0 & 0 \\[5pt]
  {\displaystyle \frac{1}{2}} & {\displaystyle \frac{1}{2}} & 0 & 0 & 0
  \\[10pt]
  {\displaystyle \frac{1}{2}} &0 &{\displaystyle \frac{1}{2}} &0 &0
  \\[10pt]
  1 & 0 & 0 & 1 & 0
  \\[5pt] \hline
   & \rule{0pt}{20pt} {\displaystyle \frac{1}{6}} & {\displaystyle \frac{1}{3}}
   & {\displaystyle \frac{1}{3}} & {\displaystyle \frac{1}{6}}
\end{array}
$$
""")

if chart_visual == 'Код с NumPy':
    st.header('Код с NumPy')
    st.write(r"""
В модуле rungeKutta функция rungeKutta() реализует решение задачи Коши 
для системы ОДУ методом Рунге-Кутта четвертого порядка.
""")
    st.subheader("Модуль rungeKutta")
    code = '''
import numpy as np
def rungeKutta(f, t0, y0, tEnd, tau):
  """
  Решите задачу о начальных значениях y' = f(t,y) 
  методом Рунге-Кутты 4-го порядка.
  t0, y0 - начальные условия,
  tEnd - конечное значение t,
  tau - шаг.
  """
  def increment(f, t, y, tau):
    k0 = tau * f(t,y)
    k1 = tau * f(t + tau/2, y + k0/2)
    k2 = tau * f(t + tau/2, y + k1/2)
    k3 = tau * f(t + tau/2, y + k2)
    return (k0 + 2. * k1 + 2. * k2 + k3) / 6.
  t = []
  y = []
  t.append(t0)
  y.append(y0) 
  while t0 < tEnd:
    tau = min(tau, tEnd - t0)
    y0 = y0 + increment(f, t0, y0, tau)
    t0 = t0 + tau 
    t.append(t0) 
    y.append(y0)
  return np.array(t), np.array(y)
'''
    st.code(code, language='python')

    st.write(r"""

  При приближенном решении модельной задачи Коши для уравнения второго порядка 
  сначала переходим от одного уравнения второго порядка к системе из двух уравнений

  $\begin{aligned}
  \frac{d y_1 }{dt} = y_2,
  \quad \frac{d y_2 }{dt} = - \sin(y_1),
  \quad 0 < t < 4 \pi ,
\end{aligned}$

Решение задачи при заданном шаге интегрирования $\tau$ обеспечивает следую­щая программа.

    """)
    st.subheader("Основной код")
    st.write(r"Приближенное решение задачи при $\tau$ = 0.25")
    code = '''
import numpy as np
import math as mt
import matplotlib.pyplot as plt 
from rungeKutta import rungeKutta 
  
def f(t, y):
  f = np.zeros((2),'float')
  f[0] = y[1]
  f[1] = - mt.sin(y [0])
  return f

t0 = int(t[0]) * np.pi
tEnd = int(t[1]) * np.pi
y0 = np.array([1., 0.])
tau = 0.25
t, y = rungeKutta(f, t0, y0, tEnd, tau)
for n in range(0, 2):
  r = y[:, n]
  st1 = "$y$" 
  sg = "-"
  if n == 1:
    st1 = "$\\frac{d y}{dt}$"
    sg = "--"

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
  t = st.slider(' ', -10 * np.pi, 10 * np.pi, (0 * np.pi, 4 * np.pi), np.pi)

  st.write(r"""Шаг интегрирования:""")
  tau_1 = st.slider('', 0.01, 1.0, 0.025)

  import numpy as np
  def rungeKutta(f, t0, y0, tEnd, tau):
    def increment(f, t, y, tau):
      k0 = tau * f(t,y)
      k1 = tau * f(t + tau/2, y + k0/2)
      k2 = tau * f(t + tau/2, y + k1/2)
      k3 = tau * f(t + tau/2, y + k2)
      return (k0 + 2. * k1 + 2. * k2 + k3) / 6.
    t = []
    y = []
    t.append(t0)
    y.append(y0) 
    while t0 < tEnd:
      tau = min(tau, tEnd - t0)
      y0 = y0 + increment(f, t0, y0, tau)
      t0 = t0 + tau 
      t.append(t0) 
      y.append(y0)
    return np.array(t), np.array(y)
  
  time_0 = time.time() 
  
  def f(t, y):
    f = np.zeros((2),'float')
    f[0] = y[1]
    f[1] = - mt.sin(y [0])
    return f

  t0 = t[0]
  tEnd = t[1]
#  t0 = 0
#  tEnd = 4 * np.pi
  y0 = np.array([1., 0.])
  tau = tau_1
  t_numpy, y_numpy = rungeKutta(f, t0, y0, tEnd, tau)

  fig, ax = plt.subplots()

  for n in range(0, 2):
    r = y_numpy[:, n]
    st1 = "$y$" 
    sg = "-"

    if n == 1:
      st1 = "$\\frac{d y}{dt}$"
      sg = "--"

    ax.plot(t_numpy, r, sg, label = st1)
  ax.legend(loc = 0)
  ax.set_xlabel('$t$')
  ax.grid()
  st.pyplot(fig)

  time_numpy = time.time() - time_0
  st.write ('Время выполнения кода NumPy = ', time_numpy)

  t_ol = np.transpose(t_numpy)
  y_new = np.transpose(y_numpy)
  
  mat = np.vstack((t_ol, y_new))
  mat1 = np.transpose(mat)
  df = pd.DataFrame(
    mat1,
    columns=('t', 'y (NumPy)', 'dy/dt (NumPy)'))  
  
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

    st.write(r"Приближенное решение задачи при $\tau$ = 0.25")
    code = '''
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

def f(t, y):
  f = np.zeros((2),'float')
  f[0] = y[1]
  f[1] = - mt.sin(y [0])
  return f
  
t0 = 0.0
tEnd = 4.*np.pi
t1 = np.array([t0, tEnd])
y0 = np.array([1., 0.])
tau = 0.25

sol = solve_ivp(f, t1, y0, first_step = tau, max_step = tau)

# для графика
t = sol.t
y = np.transpose(sol.y)
fig, ax = plt.subplots()
for n in range(0, 2):
  r = y[:, n]
  st1 = "$y$" 
  sg = "-"
  if n == 1:
    st1 = "$\\frac{d y}{dt}$"
    sg = "--"
  ax.plot(t, r, sg, label = st1)
ax.legend(loc = 0)
ax.set_xlabel('$t$')
ax.grid()
st.pyplot(fig)
    '''
    st.code(code, language='python')

if chart_visual == 'Параметрические расчеты со SciPy':
  st.header('Параметрические расчеты со SciPy')

  st.write(r"""Интервал решения задачи:""")
  t = st.slider(' ', -10 * np.pi, 10 * np.pi, (0 * np.pi, 4 * np.pi), np.pi)

  st.write(r"""Шаг интегрирования:""")
  tau_1 = st.slider('', 0.01, 1.0, 0.025)

  from scipy.integrate import solve_ivp
  import time 
  time_1 = time.time()

  def f(t, y):
    f = np.zeros((2),'float')
    f[0] = y[1]
    f[1] = - mt.sin(y [0])
    return f
  
  t0 = int(t[0])
  tEnd = int(t[1])
  t1 = np.array([t0, tEnd])
  y0 = np.array([1., 0.])
  tau = tau_1

  sol = solve_ivp(f, t1, y0, first_step = tau, max_step = tau)

  t = sol.t
  y = np.transpose(sol.y)

  fig, ax = plt.subplots()
  for n in range(0, 2):
    r = y[:, n]
    st1 = "$y$" 
    sg = "-"

    if n == 1:
      st1 = "$\\frac{d y}{dt}$"
      sg = "--"

    ax.plot(t, r, sg, label = st1)
  ax.legend(loc = 0)
  ax.set_xlabel('$t$')
  ax.grid()
  st.pyplot(fig)

  t_scipy = sol.t
  y_scipy = np.transpose(sol.y)
  time_scipy = time.time() - time_1
  st.write ('Время выполнения кода SciPy = ', time_scipy)

  t_ol = np.transpose(t_scipy)

  mat = np.vstack((t_ol, sol.y))
  mat1 = np.transpose(mat)
  df = pd.DataFrame(
    mat1,
    columns=('t', 'y (SciPy)', 'dy/dt (SciPy)'))  
  
  st.table(df)

if chart_visual == 'Сравнение':
  st.header('Сравнение')

  st.write(r"""Шаг интегрирования:""")
  tau_1 = st.slider('', 0.01, 1.0, 0.025)

  import numpy as np
  def rungeKutta(f, t0, y0, tEnd, tau):
    def increment(f, t, y, tau):
      k0 = tau * f(t,y)
      k1 = tau * f(t + tau/2, y + k0/2)
      k2 = tau * f(t + tau/2, y + k1/2)
      k3 = tau * f(t + tau/2, y + k2)
      return (k0 + 2. * k1 + 2. * k2 + k3) / 6.
    t = []
    y = []
    t.append(t0)
    y.append(y0) 
    while t0 < tEnd:
      tau = min(tau, tEnd - t0)
      y0 = y0 + increment(f, t0, y0, tau)
      t0 = t0 + tau 
      t.append(t0) 
      y.append(y0)
    return np.array(t), np.array(y)
  
  import math as mt
  import time 
  time_0 = time.time() 
  
  def f(t, y):
    f = np.zeros((2),'float')
    f[0] = y[1]
    f[1] = - mt.sin(y [0])
    return f

  t0 = 0
  tEnd = 4 * np.pi
  y0 = np.array([1., 0.])
  tau = tau_1
  t_numpy, y_numpy = rungeKutta(f, t0, y0, tEnd, tau)

  time_numpy = time.time() - time_0
  
  st.write ('Время выполнения кода NumPy = ', time_numpy)
  #st.write(t_numpy)
  #st.write(y_numpy)

  # Мой код
  from scipy.integrate import solve_ivp
  time_1 = time.time()

  t1 = np.array([t0, tEnd])

  sol = solve_ivp(f, t1, y0, first_step = tau_1, max_step = tau_1)

  t_scipy = sol.t
  y_scipy = np.transpose(sol.y)
  time_scipy = time.time() - time_1
  st.write ('Время выполнения кода SciPy = ', time_scipy)

  t_ol = np.transpose(t_numpy)
  y_new = np.transpose(y_numpy)
  y_del = y_new - sol.y
  
  mat = np.vstack((t_ol, y_new, sol.y, y_del))
  mat1 = np.transpose(mat)
  df = pd.DataFrame(
    mat1,
    columns=('t', 'y (NumPy)', 'dy/dt (NumPy)', 'y (SciPy)', 'dy/dt (SciPy)', 'Сравнение y', 'Сравнение dy/dt'))  
  
  st.table(df)

if chart_visual == 'Спасибо':
    from PIL import Image
    image = Image.open('13.1.png')
    st.image(image)
