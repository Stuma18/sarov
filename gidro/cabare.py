import numpy as np
import sympy.parsing.sympy_parser as sympy
from sympy.abc import x
import matplotlib.pyplot as plt
from scipy.constants import g 
import streamlit as st
import time

st.subheader("""Выберете параметры для постороения графика: """)

fun_1 = st.slider('Высота воды слева', 1, 10, (2), 1)
fun_2 = st.slider('Высота воды справа', 1, 10, (1), 1)
funu_1 = st.slider('Скорость воды слева', -10, 10, (2), 1)
funu_2 = st.slider('Скорость воды справа', -10, 10, (1), 1)
fun = [str(fun_1), str(fun_2)]
funu = [str(funu_1), str(funu_2)]

Nx = st.slider('Колличество точек x ', 101, 1001, (101), 2)
CFL = st.slider('Число Куранта ', 0.1, 1.0, (0.2), 0.1)
left = st.slider('Скорость воды на крайней точке слева ', -10, 10, (0), 1)
right = st.slider('Скорость воды на крайней точке справа ', -10, 10, (0), 1)

#fun = [str(fun[0]), str(fun[1])]
#funu = [str(funu[0]), str(funu[1])]


# Подготовка начальных параметров
def get_parameters(funcs, period, X):
    a = []
    j = 0
    for i in np.arange(len(funcs)):
        #записываем строку как функцию
        sys = sympy.parse_expr(funcs[i])
        xi = period[i]
        while xi < period[i+1]:
            a.append(sys.subs(x,xi).n())
            j = j+1
            xi = X[j]
    #включаем последнюю точку [)[)[]
    a.append(sys.subs(x,period[-1]).n())
    return np.array(a, dtype=np.float64)

#иттерация метода КАБАРЭ
def Cabaret(h,u,CFL,tmax,dx,t, left, right):
    
    if (t > tmax):
        return
    tau = np.min((CFL * dx)/(np.abs(u)+np.sqrt(g*h)))
    
    #фаза 1 Предиктор
    h_c_1_2, u_c_1_2 = stage_1_3(h[::2], u[::2], h[1::2], u[1::2], tau, dx)
    
    #фаза 2 (инварианты Риманна) Генерация потоков
    R, Q = R_Q(h, u)
    R_c_1_2, Q_c_1_2 = R_Q(h_c_1_2, u_c_1_2)
    R_,Q_ = stage_2(R,Q,R_c_1_2,Q_c_1_2)
    
    #переход обратно к u, h (учет граничных условий) (2,5)
    Q_ = np.append(Q_, 2 * right - R_[-1])
    R_ = np.insert(R_,0, 2 * left - Q_[0])
    hn = ((R_ - Q_)**2)/(16*g)
    un = (R_+ Q_)/2
    
    #фаза 3 (тоже самое, что и фаза 1, по из слоя n+1/2 в n+1) (зеркально) Корректор
    h_c,u_c= stage_1_3(hn, un, h_c_1_2, u_c_1_2, tau, dx)
    u = np.zeros(len(h))
    h = np.zeros(len(h))
    #собираем массивы через один
    h[::2] = hn
    h[1::2] = h_c
    u[::2] = un
    u[1::2] = u_c
    #переходим на слой по времени n+1
    t = t + tau
    return t, h, u  

def stage_1_3(h,u,h_c,u_c,tau,dx):
    h_c_1_2 = h_c[::] + tau/(2 * dx) *(h[:-1:]*u[:-1:] - h[1::]*u[1::])   
    u_c_1_2 = (h_c[::]*u_c[::]+(h[:-1:]*(u[:-1:])**2-h[1::]*(u[1::])**2+(g/2)*(h[:-1:]-h[1::])*(h[:-1:]+h[1::]))* tau/(2 * dx))/h_c_1_2 
    return h_c_1_2, u_c_1_2

def stage_2(R, Q, R_c_1_2, Q_c_1_2):
    R_ = 2 * R_c_1_2 - R[:-1:2]
    Q_ = 2 * Q_c_1_2 - Q[ 2::2]
    
    #коррекция (чтобы не было выпадов)
    for i in np.arange(len(R_c_1_2)):
        maxR = np.max((R[2*i],R[2*i+1],R[2*i + 2]))
        minR = np.min((R[2*i],R[2*i+1],R[2*i + 2]))
        minQ = np.min((Q[2*i],Q[2*i+1],Q[2*i + 2]))
        maxQ = np.max((Q[2*i],Q[2*i+1],Q[2*i + 2]))
        if R_[i] < minR:
            R_[i] = minR
        elif R_[i] > maxR:
            R_[i] = maxR
        if Q_[i] < minQ:
            Q_[i] = minQ
        elif Q_[i] > maxQ:
            Q_[i] = maxQ
    return R_, Q_

#инварианты Риммана 
def R_Q(h, u):
    return u + 2*np.sqrt(g*h), u - 2*np.sqrt(g*h)    

#высота воды
#fun = ["2","5","2"]
periods = [0, 2, 10]
#колличество точек x (более точный график)
#Nx = 101
#скорость воды
#funu = ["0","0"]
periodus = [periods[0], 5,periods[-1]]
#область x
X = np.linspace(periods[0], periods[-1], Nx)
#передаем параметры
h = get_parameters(fun, periods, X)
u = get_parameters(funu, periodus, X)

#plt.plot(X, h)
#plt.plot([0,0,10,10],[3,0.1,0.1,3], c = 'black')
#plt.ylim((0,5))
#plt.show()
t = 0.0
j = 0

chart_visual = st.sidebar.radio('Содержание',  
    ('график', 'показать график скорости', 'показать график высоты'))

if chart_visual == 'график':        

    fig, ax = plt.subplots()
    ax.plot(X, h)
    ax.plot([periods[0],periodus[0],periods[-1],periodus[-1]],[10,0.1,0.1,10], c = 'black')
    ax.set_xlabel('$x$')
    ax.set_xlabel('$h$')
    ax.set_ylim([0, 11])
    st.pyplot(fig)
    ax.clear()

    t = 0.0
    j = 0
    #число Куррента (должноо быть меньше 1, но лчше около 0.2)
    #CFL = 0.2
    #максиальное время
    tmax = 10
    #граничные точки (скорость воды на краях, если 0, то бассейн)
    #left = 0
    #right = 0

    while t < tmax:
        t, h, u = Cabaret(h, u, CFL, tmax, X[2]-X[0], t, left, right)
        #выдаем каждые 5 точек на графике
        if j%5 == 0:
            #plt.plot(X, h) высота
            #plt.plot(X, u) скорость
            #plt.plot([0,0,10,10],[3,0.1,0.1,3], c = 'black')
            #plt.show()
            ax.plot(X, h)
            # бассейн
            ax.plot([periods[0],periodus[0],periods[-1],periodus[-1]],[10,0.1,0.1,10], c = 'black')
            #plt.show()
            st.pyplot(fig)
            ax.clear()
        j += 1

if chart_visual == 'показать график скорости':
    fig, ax = plt.subplots()
    
    tmax = 1000

    ax.set_ylim(-10, 11)
    line, = ax.plot(X, u)
    the_plot = st.pyplot(plt)

    def init():
        global t, h, u
        t, h, u = Cabaret(h,u,CFL,tmax,X[2]-X[0],t,left,right)
        line.set_ydata(u)

    def animate(i):
        global t, h, u
        t, h, u = Cabaret(h,u,CFL,tmax,X[2]-X[0],t,left,right)
        line.set_ydata(u)
        the_plot.pyplot(plt)

    init()
    for i in range(100000):
        animate(i)
        time.sleep(0.01)
        
#if st.button('показать график высоты'):
if chart_visual == 'показать график высоты':
    fig, ax = plt.subplots()
    # бассейн
    ax.plot([periods[0],periodus[0],periods[-1],periodus[-1]],[10,0.1,0.1,10], c = 'black')
    
    tmax = 1000

    ax.set_ylim(0, 11)
    line, = ax.plot(X, h)
    the_plot = st.pyplot(plt)

    def init():
        global t, h, u
        t, h, u = Cabaret(h,u,CFL,tmax,X[2]-X[0],t,left,right)
        line.set_ydata(h)

    def animate(i):
        global t, h, u
        t, h, u = Cabaret(h,u,CFL,tmax,X[2]-X[0],t,left,right)
        line.set_ydata(h)
        the_plot.pyplot(plt)

    init()
    for i in range(100000):
        animate(i)
        time.sleep(0.01)
