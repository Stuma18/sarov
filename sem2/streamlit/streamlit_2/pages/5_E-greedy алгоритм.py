import streamlit as st

st.set_page_config(page_title = "E-greedy алгоритм")

st.sidebar.success("E-greedy алгоритм")

st.write("# E-greedy алгоритм")

st.markdown(
        """
E-greedy алгоритм — это один из наиболее популярных алгоритмов 
для решения дилеммы исследования-использования в машинном обучении. 


Более формально, если у нас есть множество действий $A$ и оценочные функции $q(a)$, 
которые оценивают среднюю ожидаемую награду при действии $a$, то на каждом шаге времени 
$t$ мы выбираем действие для выполнения как $$a_{t}$$ со следующей вероятностью:
""")

st.markdown(
        """
$$
P(a_{t})=(1-\epsilon)/N+\epsilon*(a_{t}=argmax(q(a)))
$$

""")

st.markdown(
        """
## Работа ε-greedy алгоритма
""")

from PIL import Image
image = Image.open('7.png')
st.image(image)