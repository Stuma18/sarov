
import streamlit as st
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
from prettytable import PrettyTable
from toolz import partition
from pages.funk.funk_eps import *

st.write("# Моделирование работы Эпсилон-жадного алгоритма")

eps = st.sidebar.slider('Эпсилон:', 0.01, 0.2, 0.1)
N_takt = st.sidebar.slider('Количество тактов:', 100, 400, 100, step = 10)

c_1_0 = st.sidebar.slider('Конверсия баннера 0 изначальная:', 0.01, 0.02, 0.01, step = 0.001)
c_1_1 = st.sidebar.slider('Конверсия баннера 1 изначальная:', 0.01, 0.02, 0.012, step = 0.001)
c_1_2 = st.sidebar.slider('Конверсия баннера 2 изначальная:', 0.01, 0.02, 0.014, step = 0.001)


N_takt_new = st.sidebar.slider('Номер такта после которого изменяется конверсия:', 100, int(N_takt/2), N_takt, step = 10)

c_2_0 = st.sidebar.slider('Конверсия баннера 0 после изменения:', 0.01, 0.02, 0.014, step = 0.001, key = 111)
c_2_1 = st.sidebar.slider('Конверсия баннера 1 после изменения:', 0.01, 0.02, 0.012, step = 0.001, key = 112)
c_2_2 = st.sidebar.slider('Конверсия баннера 2 после изменения:', 0.01, 0.02, 0.011, step = 0.001, key = 113)

st.sidebar.write("\n")
st.sidebar.write("\n")
st.sidebar.write("\n")
st.sidebar.write("\n")

conversion_1 = [c_1_0, c_1_1, c_1_2]
conversion_2 = [c_2_0, c_2_1, c_2_2]
#conversion_2 = [, c_2_1, c_2_2]

if st.button('Run'):
    eps_greedy_new (eps, N_takt, N_takt_new, conversion_1, conversion_2)